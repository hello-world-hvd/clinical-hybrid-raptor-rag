from src.generation.llm_answer import answer_with_llm
from src.retrieval.hybrid_retriever import run_query, parse_args, build_retriever
from src.retrieval.index_raptor import BM25Index

def main():
    args = parse_args()

    retriever = build_retriever(args)

    results = run_query(
        retriever,
        args,
        args.query,
    )

    answer = answer_with_llm(
        query=args.query,
        results=results,
        backend="cohere",
        qwen_model="Qwen/Qwen2.5-3B-Instruct",
        cohere_model="command-r-plus-08-2024",
        max_context_items=10,
    )

    print("\n=== ANSWER ===\n")
    print(answer)


if __name__ == "__main__":
    main()