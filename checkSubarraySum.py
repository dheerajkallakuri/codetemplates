def checkSubarraySum(nums, k):
        n = len(nums)
        if n < 2:
            return False
        for length in range(2,n+1):
            for i in range(n-length+1):
                subarr=nums[i:i+length]
                reqsum=sum(subarr)
                if reqsum % k == 0:
                    return True
        return False



def checkSubarraySum_optmise(nums, k):
    if len(nums) < 2:
        return False

    seen_mods = {0: -1}
    mod = 0
    
    for i, num in enumerate(nums):
        mod = (mod + num) % k if k != 0 else mod + num  # Handle k == 0 case
        if mod in seen_mods:
            if i - seen_mods[mod] > 1:
                return True
        else:
            seen_mods[mod] = i
    
    return False

nums=[23,2,6,4,7]
print(checkSubarraySum(nums, 13))
print(checkSubarraySum_optmise(nums, 13))