from utils import init_args
from runnable import Runnable

if __name__ == "__main__":
    args = init_args()
    r = Runnable(args)
    s = r.run()
    print(s)