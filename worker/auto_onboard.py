import sys
import json
import asyncio
import asyncpg
from config import (
    DATABASE_URL,
    SUPER_MODEL,
    validate_env,
    get_nebius_client,
    get_tavily_client,
    normalize_and_verify_domain,
    safe_parse_json,
)


def ensure_str(val, default: str = "") -> str:
    """
    Safely coerces any data type (list, dict, None, int) returned by LLMs
    into a clean string for PostgreSQL text columns.
    """
    if val is None:
        return default
    if isinstance(val, str):
        cleaned = val.strip()
        return cleaned if cleaned else default
    if isinstance(val, (list, tuple, set)):
        items = [ensure_str(item, "").strip() for item in val if item is not None]
        items = [i for i in items if i]
        return "\n".join(f"- {i}" if not i.startswith("- ") else i for i in items) if items else default
    if isinstance(val, dict):
        return "\n".join(f"- {k}: {ensure_str(v, '')}" for k, v in val.items())
    return str(val).strip()


async def auto_onboard_client(raw_url_or_domain: str):
    validate_env()

    print(f"\n[0/4] Validating website: {raw_url_or_domain}...")
    try:
        client_domain = await asyncio.to_thread(normalize_and_verify_domain, raw_url_or_domain)
        print(f"  -> Verified live domain: {client_domain}")
    except ValueError as e:
        print(f"[ERROR] {str(e)}")
        return None

    tavily_client = get_tavily_client()
    nebius_client = get_nebius_client()

    # Stage A: Deep Crawl the Client's Own Website for BI Profile & Product Lineup
    print(f"\n[1/4] Deep-crawling client business & product portfolio ({client_domain})...")
    client_search = await asyncio.to_thread(
        tavily_client.search,
        query=f"site:{client_domain} products services about pricing solutions",
        search_depth="advanced",
        max_results=4
    )

    client_urls = [r["url"] for r in client_search.get("results", [])]
    client_context = "\n".join([f"- {r['title']}: {r['content']}" for r in client_search.get("results", [])])

    # Fallback if site: search was too sparse (e.g., SPAs, newly registered domains, or restrictive robots.txt)
    if len(client_context) < 250:
        fallback_search = await asyncio.to_thread(
            tavily_client.search,
            query=f'"{client_domain}" company products services pricing solutions about',
            search_depth="advanced",
            max_results=4
        )
        for r in fallback_search.get("results", []):
            client_context += f"\n- {r['title']}: {r['content']}"
            if r.get("url") and r["url"] not in client_urls:
                client_urls.append(r["url"])

    # Always ensure homepage is extracted so we have verified company copy
    homepage_url = f"https://{client_domain}"
    urls_to_extract = [u for u in client_urls[:2]]
    if homepage_url not in urls_to_extract and len(urls_to_extract) < 3:
        urls_to_extract.append(homepage_url)

    if urls_to_extract:
        try:
            extracted = await asyncio.to_thread(tavily_client.extract, urls=urls_to_extract)
            for item in extracted.get("results", []):
                raw = item.get("raw_content", "")
                if raw:
                    client_context += f"\n\nPage Content ({item['url']}):\n{raw[:3000]}"
        except Exception:
            pass

    print(f"[2/4] Extracting BI company profile & product lineup via {SUPER_MODEL}...")
    client_prompt = f"""Target Business Domain: {client_domain}

Live Website Context:
{client_context[:5000]}

Task:
Analyze this business for the 'opeN' Business Intelligence platform.
Extract its strategic profile AND a curated portfolio of 6 to 8 core products/services.

Product Selection Strategy:
- Include 3-4 Flagship / High-Revenue Core Offerings (the company's primary revenue drivers or most famous products).
- Include 2-3 Strategic Growth / Contested Offerings (mid-tier, high-growth, or emerging products facing fierce competition, where competitive displacement strategy is crucial).
- Do NOT extract minor accessories, local/seasonal variants, or long-tail items. Cap the list at 6 to 8 products max.

Broad Category Standardization Rule:
- Group products under 3 to 5 BROAD, standardized macro-categories (e.g. 'Carbonated Soft Drinks', 'Bottled Water & Hydration', 'Payment Processing', 'Subscription & Billing').
- Do NOT create narrow or hyper-specific micro-categories (e.g. do NOT use 'Diet Cola', 'Lemon-Lime Soda', or 'Zero-Sugar Soda' — all of these MUST share the exact same broad category 'Carbonated Soft Drinks').

Return ONLY raw valid JSON (no markdown fences) matching this exact schema:
{{
  "name": "Official Company Name",
  "industry": "Primary Industry / Market Category",
  "value_proposition": "Concise 1-2 sentence core value proposition",
  "key_differentiators": "Top 2-3 strategic or market differentiators",
  "pricing_model": "Summary of how they monetize or price their offerings",
  "products": [
    {{
      "product_name": "Specific Product or Brand Name",
      "product_type": "Broad standardized category (max 3-5 unique across entire list, e.g. 'Carbonated Soft Drinks')",
      "portfolio_role": "flagship_core or strategic_growth",
      "description": "Clear summary of what this product is and its core capabilities",
      "target_audience": "Primary customer or market segment"
    }}
  ]
}}"""

    profile_res = await nebius_client.chat.completions.create(
        model=SUPER_MODEL,
        messages=[
            {"role": "system", "content": "You are an objective Business Intelligence analyst. Output raw valid JSON only."},
            {"role": "user", "content": client_prompt}
        ],
        temperature=0.0
    )

    client_profile = safe_parse_json(profile_res)
    company_name = ensure_str(client_profile.get("name") or client_profile.get("company_name"), client_domain)
    industry = ensure_str(client_profile.get("industry"), "Enterprise / Commercial")
    
    # Robust product list parsing handling lists, dicts, or strings
    raw_prods = client_profile.get("products", [])
    products_list = []
    if isinstance(raw_prods, list):
        for p in raw_prods:
            if isinstance(p, dict):
                products_list.append(p)
            elif isinstance(p, str):
                products_list.append({"product_name": p, "product_type": "Core Offering", "description": p})
    elif isinstance(raw_prods, dict):
        for k, v in raw_prods.items():
            if isinstance(v, dict):
                products_list.append({"product_name": k, **v})
            else:
                products_list.append({"product_name": k, "product_type": "Core Offering", "description": str(v)})

    if not products_list:
        products_list = [{
            "product_name": f"{company_name} Core Offering",
            "product_type": industry,
            "portfolio_role": "flagship_core",
            "description": ensure_str(client_profile.get("value_proposition"), "Primary business solution"),
            "target_audience": "General Market"
        }]

    print(f"  -> Profiled '{company_name}' ({industry}) with {len(products_list)} core product(s).")

    # Stage B: Market Share Search to Discover Consensus Top 5 Direct Competitors
    print(f"[3/4] Auditing market share landscape for top 5 competitors of {company_name} ({client_domain})...")
    market_search = await asyncio.to_thread(
        tavily_client.search,
        query=f"top direct competitors of {company_name} {client_domain} {industry} market share revenue",
        search_depth="advanced",
        max_results=5
    )

    market_context = "\n".join([f"- {r['title']}: {r['content']}" for r in market_search.get("results", [])])

    comp_prompt = f"""Target Business: {company_name} ({client_domain})
Industry: {industry}
Core Products: {json.dumps(products_list)}

Live Market Intelligence:
{market_context[:4500]}

Task:
Identify the top 5 direct global competitors to {company_name}, strictly ranked by primary product overlap and market share.
Do NOT include {company_name} ({client_domain}) itself.

Return ONLY raw valid JSON (no markdown code fences) matching this exact schema:
{{
  "competitors": [
    {{"name": "Competitor 1 Name", "domain": "competitor1.com"}},
    {{"name": "Competitor 2 Name", "domain": "competitor2.com"}},
    {{"name": "Competitor 3 Name", "domain": "competitor3.com"}},
    {{"name": "Competitor 4 Name", "domain": "competitor4.com"}},
    {{"name": "Competitor 5 Name", "domain": "competitor5.com"}}
  ]
}}"""

    comp_res = await nebius_client.chat.completions.create(
        model=SUPER_MODEL,
        messages=[
            {"role": "system", "content": "You are a market intelligence researcher. Output raw valid JSON only."},
            {"role": "user", "content": comp_prompt}
        ],
        temperature=0.0
    )

    comp_data = safe_parse_json(comp_res)
    raw_comps = comp_data.get("competitors", [])
    competitors = []
    if isinstance(raw_comps, list):
        for c in raw_comps:
            if isinstance(c, dict):
                competitors.append(c)
            elif isinstance(c, str):
                competitors.append({"name": c, "domain": c})
    elif isinstance(raw_comps, dict):
        for k, v in raw_comps.items():
            competitors.append({"name": k, "domain": str(v)})

    # Stage C: Seed Supabase Database (Wiping old session cleanly)
    print("[4/4] Committing BI baseline, product catalog, and top 5 competitors to Supabase...")
    pool = await asyncpg.create_pool(DATABASE_URL, min_size=1, max_size=3)

    saved_competitors = []
    async with pool.acquire() as conn:
        async with conn.transaction():
            # 1. Wipe previous session data (cascades to products, snapshots, staging, battlecards)
            await conn.execute("DELETE FROM company_profile")
            await conn.execute("DELETE FROM competitors")

            # 2. Insert new Company Profile with strict string coercion
            company_id = await conn.fetchval(
                """
                INSERT INTO company_profile (domain, name, industry, value_proposition, key_differentiators, pricing_model)
                VALUES ($1, $2, $3, $4, $5, $6)
                RETURNING id
                """,
                client_domain,
                company_name,
                industry,
                ensure_str(client_profile.get("value_proposition"), "Not specified"),
                ensure_str(client_profile.get("key_differentiators"), "Not specified"),
                ensure_str(client_profile.get("pricing_model"), "Not specified")
            )

            # 3. Insert curated strategic product portfolio (capped at 8 products max)
            for prod in products_list[:8]:
                p_name = ensure_str(prod.get("product_name") or prod.get("name"), "Core Offering")
                p_type = ensure_str(prod.get("product_type") or prod.get("category"), "General Product")
                p_role = ensure_str(prod.get("portfolio_role") or prod.get("role"), "flagship_core")
                if p_role not in ("flagship_core", "strategic_growth"):
                    p_role = "strategic_growth" if "growth" in p_role.lower() else "flagship_core"
                p_desc = ensure_str(prod.get("description") or prod.get("summary"), "")
                p_target = ensure_str(prod.get("target_audience") or prod.get("audience"), "General Market")

                await conn.execute(
                    """
                    INSERT INTO products (company_id, product_name, product_type, portfolio_role, description, target_audience)
                    VALUES ($1, $2, $3, $4, $5, $6)
                    """,
                    company_id,
                    p_name,
                    p_type,
                    p_role,
                    p_desc,
                    p_target
                )

            # 4. Insert Top 5 Competitors
            for comp in competitors[:5]:
                c_name = ensure_str(comp.get("name") or comp.get("company_name"), "")
                raw_comp_domain = ensure_str(comp.get("domain") or comp.get("website") or comp.get("url"), "")
                if not c_name or not raw_comp_domain:
                    continue

                try:
                    c_domain = await asyncio.to_thread(normalize_and_verify_domain, raw_comp_domain)
                except ValueError:
                    # Fallback basic cleanup if a competitor's secondary domain doesn't resolve immediately
                    c_domain = raw_comp_domain.strip().lower().replace("https://", "").replace("http://", "").split("/")[0]
                    if c_domain.startswith("www."):
                        c_domain = c_domain[4:]

                if not c_domain or c_domain == client_domain:
                    continue

                pricing_url = f"https://{c_domain}/pricing"
                changelog_url = f"https://{c_domain}/changelog"

                await conn.execute(
                    """
                    INSERT INTO competitors (name, domain, pricing_url, changelog_url, is_active)
                    VALUES ($1, $2, $3, $4, TRUE)
                    ON CONFLICT (domain) DO UPDATE SET is_active = TRUE
                    """,
                    c_name, c_domain, pricing_url, changelog_url
                )
                print(f"  + Seeded Competitor: {c_name} ({c_domain})")
                saved_competitors.append({"name": c_name, "domain": c_domain})

    await pool.close()
    print(f"[SUCCESS] Phase 0 Onboarding complete for {company_name} ({len(saved_competitors)} competitors active)!\n")

    return {
        "client_name": company_name,
        "domain": client_domain,
        "industry": industry,
        "products_count": len(products_list),
        "competitors": saved_competitors
    }


if __name__ == "__main__":
    if len(sys.argv) > 1:
        target = sys.argv[1]
    else:
        target = input("Enter client URL or domain (e.g., https://www.coca-cola.com or coca-cola.com): ").strip()

    if target:
        asyncio.run(auto_onboard_client(target))