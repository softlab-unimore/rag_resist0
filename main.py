from utils import init_args
from runnable import Runnable
from evaluation import Evaluator

if __name__ == "__main__":
    args = init_args()
    runner = Runnable(args)
    evaluator = Evaluator()

    if args["use_llama"]:
        results = runner.run_with_llama()
    else:
        results = runner.run()
    
    scores = evaluator.evaluate(results)
    print(scores)