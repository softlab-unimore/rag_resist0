from utils import init_args
from runnable import Runnable

if __name__ == "__main__":
    args = init_args()
    runner = Runnable(args)

    if args["use_llama"]:
        results = runner.run_with_llama()
    else:
        results = runner.run()

    if not args["use_ensemble"]:
        results = [r[0] for r in results]

    result_llm = runner.run_value_extraction(results)
