def minIncrementOperations(self, nums, k):
    """
    :type nums: List[int]
    :type k: int
    :rtype: int
    """
    dp1,dp2,dp3=0,0,0
    for num in nums:
        dp1,dp2,dp3=dp2,dp3,min(dp1,dp2,dp3)+max(k-num,0)
    return min(dp1,dp2,dp3)

nums = [1,1,4,2,0,4]
print(minIncrementOperations(nums))