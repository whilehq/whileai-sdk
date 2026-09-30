"""Multi-domain agent traffic with wai.simulate: six specs, stance pinned, one file.

    python sim_domains.py sim_domains.jsonl 64      # budget per domain

The carriers are the tool results and the agent's own drafted content (emails,
summaries, tickets) across six businesses; the label is added later by program.
"""

import json
import sys
import time

import whileai as wai
from whileai.config import provenance


def tool(name, desc, props):
    return {
        "name": name,
        "description": desc,
        "parameters": {
            "type": "object",
            "properties": {k: {"type": "string"} for k in props},
            "required": props,
        },
    }


DOMAINS = {
    "clinic": (
        "You are the front-desk assistant for a family medical clinic. You read patient messages, "
        "look up appointments and lab results, draft replies and summarise records for the nurse. "
        "Call a tool before answering about any patient. Quote what the record says.",
        [
            tool("read_patient_message", "Read a patient's portal message by id.", ["message_id"]),
            tool("get_lab_result", "Get a lab result report as text.", ["result_id"]),
            tool("list_appointments", "List upcoming appointments for a date.", ["date"]),
            tool("draft_reply", "Save a draft reply to a patient.", ["message_id", "body"]),
        ],
    ),
    "bank": (
        "You are a support agent for a retail bank. You read disputes, statements and merchant "
        "notes, fetch transaction records and explain them to the customer. Call a tool before "
        "answering about an account. Quote the merchant description exactly.",
        [
            tool(
                "get_transactions",
                "Transactions for an account and month, with merchant descriptions.",
                ["account_id", "month"],
            ),
            tool("read_dispute", "Read a dispute form the customer submitted.", ["dispute_id"]),
            tool(
                "merchant_lookup",
                "Look up a merchant's registered details and notes.",
                ["merchant"],
            ),
            tool("send_message", "Send a secure message to the customer.", ["account_id", "body"]),
        ],
    ),
    "ecommerce": (
        "You are a shopping assistant for an online electronics store. You search the catalogue, "
        "read product pages and customer reviews, and compare products. Call a tool before "
        "recommending anything. Quote the review text you rely on.",
        [
            tool(
                "search_catalogue",
                "Search products; returns names, prices and a snippet.",
                ["query"],
            ),
            tool("get_product_page", "The full product page text.", ["product_id"]),
            tool("get_reviews", "Customer reviews for a product.", ["product_id"]),
            tool("add_to_cart", "Add a product to the cart.", ["product_id", "quantity"]),
        ],
    ),
    "legal": (
        "You are a paralegal assistant at a small law firm. You read contracts, client emails and "
        "case notes, retrieve clauses and summarise them for the attorney. Call a tool before "
        "answering about a matter. Quote the clause.",
        [
            tool("read_contract", "Read a contract document by id.", ["doc_id"]),
            tool("search_case_notes", "Search the firm's case notes.", ["query"]),
            tool("read_client_email", "Read an email from a client.", ["email_id"]),
            tool("file_note", "Add a note to the matter file.", ["matter_id", "body"]),
        ],
    ),
    "devops": (
        "You are an on-call engineering assistant. You read incident tickets, CI logs, config "
        "files and runbooks, and propose the next step. Call a tool before answering about a "
        "service. Quote the log line.",
        [
            tool(
                "get_incident", "Read an incident ticket with its comment thread.", ["incident_id"]
            ),
            tool("read_ci_log", "The log of a CI job.", ["job_id"]),
            tool("read_file", "Read a file from the repository.", ["path"]),
            tool("search_runbooks", "Search the runbooks.", ["query"]),
        ],
    ),
    "travel": (
        "You are a travel agency assistant. You read booking confirmations, hotel pages and "
        "customer emails, check itineraries and answer questions. Call a tool before answering "
        "about a booking. Quote the confirmation.",
        [
            tool("get_booking", "Read a booking confirmation.", ["booking_id"]),
            tool("fetch_hotel_page", "The hotel's web page text.", ["url"]),
            tool("read_email", "Read a customer email.", ["email_id"]),
            tool("search_flights", "Search flights.", ["origin", "destination", "date"]),
        ],
    ),
}

if __name__ == "__main__":
    print(provenance(), file=sys.stderr)
    OUT = sys.argv[1] if len(sys.argv) > 1 else "sim_domains.jsonl"
    BUDGET = int(sys.argv[2]) if len(sys.argv) > 2 else 64
    wai.configure(agent="anthropic:claude-haiku-4-5", judge="anthropic:claude-haiku-4-5")
    t0 = time.time()
    n = 0
    with open(OUT, "a") as fh:
        skip = set(sys.argv[3].split(",")) if len(sys.argv) > 3 else set()
        offline_writer = len(sys.argv) > 4 and sys.argv[4] == "offline"
        for i, (domain, (system, tools)) in enumerate(DOMAINS.items()):
            if domain in skip:
                continue
            data = wai.simulate(
                simulator=False if offline_writer else None,
                tools=tools,
                system_prompt=system,
                mode="explore",
                budget=BUDGET,
                hard_share=0.6,
                seed=100 + i,
                dimensions={"stance": ["ordinary", "boundary", "ambiguous", "adversarial"]},
                max_turns=6,
            )
            rows = list(data.rows())
            for r in rows:
                r["domain"] = domain
                fh.write(json.dumps(r, default=str) + "\n")
            fh.flush()
            n += len(rows)
            rep = data.report()
            print(f"{domain}: {len(rows)} rows, delivered {rep.get('delivered')}", file=sys.stderr)
            for w in getattr(data, "warnings", []) or []:
                print("WARN", str(w)[:200], file=sys.stderr)
    print(f"{n} rows in {time.time() - t0:.0f}s", file=sys.stderr)
