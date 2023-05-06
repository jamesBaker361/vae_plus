import sys
sys.path.append("data_utils")
from processing_utils import *

def equal_length_test():
    big_length=5
    smaller=[_ for _ in range(1)]
    bigger = [_ for _ in range(big_length)]
    smaller, bigger = equal_length(smaller, bigger)
    assert len(bigger) == len(smaller)
    assert big_length ==len(smaller)

def equal_length_switched_test():
    big_length=5
    smaller=[_ for _ in range(1)]
    bigger = [_ for _ in range(big_length)]
    smaller, bigger = equal_length(bigger, smaller)
    assert len(bigger) == len(smaller)
    assert big_length ==len(smaller)

if __name__=='__main__':
    equal_length_test()
    equal_length_switched_test()
    print("all done :-)")