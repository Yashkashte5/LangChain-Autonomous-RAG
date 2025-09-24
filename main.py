# main.py

import argparse
from src.rag_chain import RAG
from dotenv import load_dotenv

# Load .env to get GEMINI_API_KEY
load_dotenv()

def main():
    parser = argparse.ArgumentParser(description="Autonomous RAG with LangChain + Gemini")
    subparsers = parser.add_subparsers(dest="command")

    # Build vectorstore
    build_parser = subparsers.add_parser(
        "build",
        help="Ingest all documents from data/raw into Chroma DB"
    )

    # Query vectorstore
    query_parser = subparsers.add_parser(
        "query",
        help="Ask a question to the RAG system"
    )
    query_parser.add_argument(
        "question", type=str, help="Question to ask the RAG system"
    )
    query_parser.add_argument(
        "--k", type=int, default=3, help="Number of top docs to retrieve"
    )

    args = parser.parse_args()
    rag = RAG()

    if args.command == "build":
        rag.build_vectorstore("data/raw")

    elif args.command == "query":
        answer = rag.query(args.question, top_k=args.k)
        print("\n========================================\n")
        print(answer)

    else:
        parser.print_help()


if __name__ == "__main__":
    main()


import warnings
from langchain import LangChainDeprecationWarning

warnings.filterwarnings("ignore", category=LangChainDeprecationWarning)
