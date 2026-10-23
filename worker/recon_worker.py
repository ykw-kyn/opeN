import sys
import re
import asyncio
import asyncpg
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from urllib.parse import urlparse
from config import (
    DATABASE_URL,
    validate_env,
    get_tavily_client,
)

# Concurrency limit to respect Tavily and Supabase connection pools
MAX_CONCURRENT_COMPETITORS = 3


def normalize_url(url: str) -> str:
    """Strips tracking, session, and pagination query parameters to reveal canonical URL."""
    p = urlparse(url)
    netloc = p.netloc.lower()
    if netloc.startswith("www."):
        netloc = netloc[4:]
    clean_path = p.path.rstrip("/")
    return f"{p.scheme}://{netloc}{clean_path}"


def normalize_title(title: str) -> str:
    """Normalizes titles by stripping publisher suffixes ('| Reuters', '— TradingView') and punctuation."""
    title = re.split(r"\s+[|\-—]\s+", title)[0]
    return re.sub(r"[^a-zA-Z0-9 ]", "", title.lower()).strip()


def is_outdated_source(url: str, published_date_str: str = None) -> tuple[bool, str]:
    """
    Validates recency across all intelligence sources:
    1. Checks published_date metadata if provided by Tavily (> 365 days old -> reject).
    2. Checks for historical years in URL path (e.g. /2018/, /2021/, /2023/ -> reject).
    3. Rejects SEC 10-K / annual statutory accounting disclosures.
    """
    now = datetime.now(timezone.utc)
    current_year = now.year

    if published_date_str:
        try:
            pub_date = parsedate_to_datetime(published_date_str)
            if pub_date.tzinfo is None:
                pub_date = pub_date.replace(tzinfo=timezone.utc)
            if (now - pub_date).days > 365:
                return True, f"Published date ({pub_date.strftime('%Y-%m-%d')}) is older than 1 year"
        except Exception:
            pass

    url_lower = url.lower()
    year_match = re.search(r"/(19\d\d|20[0-9]{2})/", url_lower)
    if year_match:
        url_year = int(year_match.group(1))
        if url_year < (current_year - 1):
            return True, f"URL path references historical year /{url_year}/"

    if "10-k" in url_lower or "form-10k" in url_lower or "annual-report" in url_lower:
        return True, "SEC 10-K / Annual statutory accounting report"

    return False, "Current"


async def fetch_client_product_lens(conn) -> dict:
    """
    Reads company_profile and products tables so reconnaissance queries
    target all distinct product types/categories the client offers.
    """
    company = await conn.fetchrow(
        "SELECT id, name, industry FROM company_profile LIMIT 1"
    )
    if not company:
        return {
            "client_name": "Client",
            "industry": "",
            "product_types": ["products services pricing"],
            "products_count": 0,
        }

    # Fetch ALL products for this client without arbitrary LIMITs
    product_rows = await conn.fetch(
        "SELECT product_name, product_type, description FROM products WHERE company_id = $1 ORDER BY created_at ASC",
        company["id"],
    )

    # Group and deduplicate unique product categories (preserving appearance order)
    product_types = []
    seen = set()
    for r in product_rows:
        pt = (r.get("product_type") or "").strip()
        if pt and pt.lower() not in seen:
            seen.add(pt.lower())
            product_types.append(pt)

    # Fallback to product names if product_type is not yet populated
    if not product_types:
        for r in product_rows:
            pname = (r.get("product_name") or "").strip()
            if pname and pname.lower() not in seen:
                seen.add(pname.lower())
                product_types.append(pname)

    # Ultimate fallback to industry if no products exist
    if not product_types:
        product_types = [company["industry"]] if company["industry"] else ["products services"]

    return {
        "client_name": company["name"],
        "industry": company["industry"] or "",
        "product_types": product_types,
        "products_count": len(product_rows),
    }


async def process_single_competitor(
    conn_pool,
    tavily_client,
    job_run_id,
    comp: dict,
    product_lens: dict,
    semaphore: asyncio.Semaphore,
) -> bool:
    async with semaphore:
        comp_id = comp["id"]
        comp_name = comp["name"]
        comp_domain = comp["domain"]
        industry = product_lens["industry"]
        product_types = product_lens["product_types"]

        timestamp = datetime.now(timezone.utc).strftime("%H:%M:%S")
        print(f"[{timestamp}] Starting multi-source BI recon for {comp_name} ({comp_domain})...")
        print(f"  -> Scanning across {len(product_types)} product category lens(es): {', '.join(product_types)}")

        # Hybrid Intelligence Vectors:
        # 1. Consolidated OFFICIAL_MOVE per competitor domain (high-efficiency single pass)
        recon_vectors = [
            {
                "category": "OFFICIAL_MOVE",
                "product_type": "Corporate / Multi-Category",
                "query": f'site:{comp_domain} (newsroom OR "press release" OR launch OR announcement OR "new product" OR "pricing") -\'senior notes\' -\'common stock\' -\'secondary offering\'',
                "max_results": 3,
                "time_range": "year",
                "topic": "news",
            }
        ]

        # 2. Targeted Open-Web Sweeps (EXTERNAL_NEWS & MARKET_FRICTION) per unique product category
        for pt in product_types:
            clean_pt = pt.replace("&", "OR").replace("/", " OR ").strip()

            recon_vectors.append({
                "category": "EXTERNAL_NEWS",
                "product_type": pt,
                "query": f'"{comp_name}" ({clean_pt}) (launch OR strategy OR market share OR partnership OR expansion OR revenue) -site:{comp_domain}',
                "max_results": 2,
                "time_range": "year",
                "topic": "news",
            })
            recon_vectors.append({
                "category": "MARKET_FRICTION",
                "product_type": pt,
                "query": f'"{comp_name}" ({clean_pt}) (issue OR complaint OR weakness OR controversy OR outage OR switching OR review) -site:{comp_domain}',
                "max_results": 2,
                "time_range": "year",
                "topic": "general",
            })

        staged_count = 0
        seen_clean_urls = set()
        seen_titles = set()

        # Pre-load already staged sources for this competitor to prevent duplicates across runs
        async with conn_pool.acquire() as conn:
            existing_rows = await conn.fetch(
                "SELECT source_url, raw_markdown FROM raw_recon_staging WHERE competitor_id = $1",
                comp_id,
            )
            for er in existing_rows:
                s_url = er.get("source_url")
                if s_url:
                    seen_clean_urls.add(normalize_url(s_url))
                tm = re.search(r"Source Title:\s*(.+)", er.get("raw_markdown") or "")
                if tm:
                    nt = normalize_title(tm.group(1))
                    if len(nt) >= 15:
                        seen_titles.add(nt)

        try:
            for vector in recon_vectors:
                category = vector["category"]
                pt = vector["product_type"]
                query = vector["query"]
                topic = vector.get("topic", "general")
                time_range = vector.get("time_range", "year")

                if category == "OFFICIAL_MOVE":
                    print(f"  -> [{comp_name} | {category}] Searching: {query[:85]}...")
                else:
                    print(f"  -> [{comp_name} | {category} | {pt}] Searching: {query[:85]}...")

                search_res = await asyncio.to_thread(
                    tavily_client.search,
                    query=query,
                    search_depth="advanced",
                    include_raw_content=True,
                    max_results=vector["max_results"],
                    time_range=time_range,
                    topic=topic,
                )

                results = search_res.get("results", [])

                # Fallback for OFFICIAL_MOVE if competitor's primary domain is placeholder/sparse
                if not results and category == "OFFICIAL_MOVE":
                    fallback_query = f'"{comp_name}" official (newsroom OR press release OR launch OR announcement) -\'senior notes\' -\'secondary offering\''
                    print(f"  -> [{comp_name} | OFFICIAL_MOVE] Domain sparse, trying fallback: {fallback_query[:75]}...")
                    try:
                        fb_res = await asyncio.to_thread(
                            tavily_client.search,
                            query=fallback_query,
                            search_depth="advanced",
                            include_raw_content=True,
                            max_results=vector["max_results"],
                            time_range="year",
                            topic="news",
                        )
                        results = fb_res.get("results", [])
                    except Exception:
                        pass

                if not results:
                    continue

                # Find URLs that didn't return inline raw_content and extract them
                urls_needing_extract = [
                    r["url"] for r in results if not r.get("raw_content") and r.get("url")
                ]
                extracted_map = {}
                if urls_needing_extract:
                    try:
                        ext_res = await asyncio.to_thread(
                            tavily_client.extract, urls=urls_needing_extract
                        )
                        for item in ext_res.get("results", []):
                            extracted_map[item["url"]] = item.get("raw_content", "")
                    except Exception:
                        pass

                async with conn_pool.acquire() as conn:
                    for item in results:
                        source_url = item.get("url")
                        if not source_url:
                            continue

                        # 1. Non-intelligence / Noise Filtering: Ignore job postings and social media
                        url_lower = source_url.lower()
                        if any(k in url_lower for k in ["jobs.", "careers.", "jobdetails.", "/job/", "/jobs/", "/careers/"]):
                            print(f"     [SKIP JUNK] Job posting ignored: {source_url[:65]}...")
                            continue

                        if any(k in url_lower for k in ["facebook.com", "instagram.com", "tiktok.com", "threads.net"]):
                            print(f"     [SKIP JUNK] Social media link ignored: {source_url[:65]}...")
                            continue

                        # 2. Recency & Document Quality Check (rejects >365d old items and SEC 10-K accounting filings)
                        outdated, out_reason = is_outdated_source(source_url, item.get("published_date"))
                        if outdated:
                            print(f"     [SKIP OUTDATED] {out_reason}: {source_url[:65]}...")
                            continue

                        # 3. Canonical URL Deduplication (strips pagination and tracking params)
                        clean_url = normalize_url(source_url)
                        if clean_url in seen_clean_urls:
                            print(f"     [SKIP DUP URL] Canonical URL already staged: {clean_url[:65]}...")
                            continue

                        title = item.get("title", "Untitled Source")
                        norm_title = normalize_title(title)

                        # 4. Syndicated Wire / Cross-Publisher Story Deduplication
                        if len(norm_title) >= 15 and norm_title in seen_titles:
                            print(f"     [SKIP SYNDICATED] Identical story already staged: {title[:55]}...")
                            continue

                        # Combine title, summary snippet, and full raw markdown (fallback-safe for paywalled news)
                        raw_body = (
                            item.get("raw_content")
                            or extracted_map.get(source_url)
                            or item.get("content")
                            or ""
                        )
                        snippet = item.get("content", "")

                        full_markdown = (
                            f"Source Title: {title}\n"
                            f"Source URL: {source_url}\n"
                            f"Intelligence Type: {category}\n"
                            f"Target Product Category: {pt}\n"
                            f"Key Snippet: {snippet}\n\n"
                            f"Full Extracted Content:\n{raw_body[:6000]}"
                        ).strip()

                        if len(full_markdown) < 120:
                            continue

                        await conn.execute(
                            """
                            INSERT INTO raw_recon_staging
                            (job_run_id, competitor_id, intel_category, source_url, raw_markdown, processed)
                            VALUES ($1, $2, $3, $4, $5, FALSE)
                            """,
                            job_run_id,
                            comp_id,
                            category,
                            source_url,
                            full_markdown,
                        )
                        staged_count += 1
                        seen_clean_urls.add(clean_url)
                        if len(norm_title) >= 15:
                            seen_titles.add(norm_title)

            async with conn_pool.acquire() as conn:
                await conn.execute(
                    "UPDATE competitors SET last_scanned_at = NOW() WHERE id = $1",
                    comp_id,
                )

            print(f"[SUCCESS] {comp_name}: Staged {staged_count} new intelligence item(s).")
            return True

        except Exception as e:
            print(f"[ERROR] Failed recon for {comp_name}: {str(e)}", file=sys.stderr)
            return False


async def main():
    validate_env()
    tavily_client = get_tavily_client()

    print("\nConnecting to Supabase for Phase 1 Reconnaissance...")
    pool = await asyncpg.create_pool(DATABASE_URL, min_size=2, max_size=8)

    job_run_id = None
    try:
        async with pool.acquire() as conn:
            # 1. Log the start of the job in recon_job_runs immediately
            job_run_id = await conn.fetchval(
                """
                INSERT INTO recon_job_runs (status, nebius_job_id, competitors_scanned)
                VALUES ('RUNNING', 'opeN-local-worker', 0)
                RETURNING id
                """
            )

            # 2. Load client product catalog to focus our competitor queries
            product_lens = await fetch_client_product_lens(conn)
            print(
                f"Loaded Client Lens: {product_lens['client_name']} "
                f"({product_lens['products_count']} products across {len(product_lens['product_types'])} categories) "
                f"| Industry: {product_lens['industry']}\n"
                f"Categories: {', '.join(product_lens['product_types'])}\n"
            )

            # 3. Fetch active competitors (Top 5 seeded in Phase 0)
            competitors = await conn.fetch(
                "SELECT * FROM competitors WHERE is_active = TRUE ORDER BY name ASC"
            )

        print(f"Job Run {job_run_id}: Found {len(competitors)} active competitor(s) to scan.\n")

        if not competitors:
            print("[WARN] No active competitors found. Run `python auto_onboard.py <domain>` first.")
            async with pool.acquire() as conn:
                await conn.execute(
                    """
                    UPDATE recon_job_runs
                    SET status = 'COMPLETED', competitors_scanned = 0, completed_at = NOW()
                    WHERE id = $1
                    """,
                    job_run_id,
                )
            return

        # 4. Run concurrent multi-source reconnaissance across all 5 competitors
        semaphore = asyncio.Semaphore(MAX_CONCURRENT_COMPETITORS)
        tasks = [
            process_single_competitor(
                pool, tavily_client, job_run_id, dict(comp), product_lens, semaphore
            )
            for comp in competitors
        ]
        results = await asyncio.gather(*tasks)
        success_count = sum(1 for r in results if r)

        # 5. Mark recon_job_runs as COMPLETED
        async with pool.acquire() as conn:
            await conn.execute(
                """
                UPDATE recon_job_runs
                SET status = 'COMPLETED', competitors_scanned = $1, completed_at = NOW()
                WHERE id = $2
                """,
                success_count,
                job_run_id,
            )

        print(
            f"\n[PHASE 1 COMPLETE] Job {job_run_id} finished: "
            f"{success_count}/{len(competitors)} competitors scanned."
        )

    except Exception as e:
        if job_run_id:
            async with pool.acquire() as conn:
                await conn.execute(
                    """
                    UPDATE recon_job_runs
                    SET status = 'FAILED', error_log = $1, completed_at = NOW()
                    WHERE id = $2
                    """,
                    str(e),
                    job_run_id,
                )
        raise
    finally:
        await pool.close()


if __name__ == "__main__":
    asyncio.run(main())