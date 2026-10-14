import argparse
import sys
import json
from src.core.orchestrator import AgenticWorkflowOrchestrator
from src.eval.ragas_synthetic_benchmarks import EnterpriseRAGBenchmarker
from src.guardrails.adversarial_test_cases import AdversarialSecurityTestSuite

def print_banner():
    banner = """
========================================================================
   🧠 AGENTIC AI WORKFLOW & GUARDRAILED RAG ORCHESTRATION ENGINE
   Author: Himanshu Sheta (M.Sc. Applied Computer Science)
   Core: Evaluator-Optimizer Architecture | Hybrid BM25 + Vector Retrieval
========================================================================
"""
    print(banner)

def main():
    print_banner()
    parser = argparse.ArgumentParser(description="Agentic AI CLI Interface")
    parser.add_argument("--query", type=str, default=None, help="Query for multi-agent reasoning")
    parser.add_argument("--benchmark", action="store_true", help="Run RAGAS evaluation suite")
    parser.add_argument("--security-audit", action="store_true", help="Run adversarial guardrails audit")
    args = parser.parse_args()

    orchestrator = AgenticWorkflowOrchestrator()

    if args.security_audit:
        print("[*] Running Adversarial Guardrail Security Audit...")
        suite = AdversarialSecurityTestSuite()
        report = suite.run_security_audit()
        print(f"[+] Defense Success Rate: {report['defense_success_rate']}%")
        print(json.dumps(report, indent=2))
        return

    if args.benchmark:
        print("[*] Executing RAGAS Benchmarking Suite...")
        benchmarker = EnterpriseRAGBenchmarker()
        sample_cases = [
            {
                "id": "tc_01",
                "query": "How does reciprocal rank fusion combine search results?",
                "retrieved_contexts": ["Reciprocal Rank Fusion fuses BM25 lexical rank with dense vector cosine similarity."],
                "generated_answer": "Reciprocal Rank Fusion integrates sparse keyword and dense semantic rankings to improve retrieval."
            },
            {
                "id": "tc_02",
                "query": "What are the security safeguards against prompt injection?",
                "retrieved_contexts": ["Input guardrails scan for developer overrides, jailbreaks, and sanitize sensitive PII tokens."],
                "generated_answer": "Safeguards detect prompt injection attacks and redact PII like IBANs and phone numbers."
            }
        ]
        res = benchmarker.run_benchmark_suite(sample_cases)
        print(json.dumps(res, indent=2))
        return

    query = args.query or "Explain the architecture of the Evaluator-Optimizer agentic workflow"
    print(f"[*] Processing Query: '{query}'")
    state = orchestrator.execute_workflow("cli_task_001", query)
    print("
--- Execution Output ---")
    print(f"Status: {state.status.value}")
    print(f"Route:  {state.route_selected}")
    print(f"Grounded Output:
{state.final_output}")
    print(f"Latency: {state.latency_ms} ms | Tokens: {state.tokens_consumed}")

if __name__ == "__main__":
    main()
