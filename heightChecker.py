def heightChecker(heights):
    """
    :type heights: List[int]
    :rtype: int
    """
    rh=sorted(heights)
    change_count = 0
    for i in range(len(heights)):
        if rh[i] != heights[i]:
            change_count += 1
    return change_count

print(heightChecker([1,1,4,2,1,3]))