from utils import init_args
from runnable import Runnable

if __name__ == "__main__":
    args = init_args()
    runner = Runnable(args)

    if args["embed"] and not args["query"]:
        print("Running in EMBED mode.")
        runner.run()
        print("Embedding completed successfully. Exiting.")

    if args["query"] and not args["embed"]:
        print("Running in QUERY mode.")

        if args["use_llama"]:
            results = runner.run_with_llama()
        else:
            results = runner.run()

        if not args["use_ensemble"]:
            if results is None:
                print("No results found in the vector store.")
                results = []
            else:
                results = [r[0] for r in results]

        result_llm = runner.run_value_extraction(results)
        print(result_llm)

    if args["query"] and args["embed"]:
        print("Error: You must specify either --embed or --query.")

