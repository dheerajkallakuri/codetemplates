def minSum(nums1, nums2):
    sa = sum(max(a, 1) for a in nums1)
    sb = sum(max(b, 1) for b in nums2)
    if sa < sb and nums1.count(0) == 0: return -1
    if sa > sb and nums2.count(0) == 0: return -1
    return max(sa, sb)

def minSum(self, nums1, nums2):
    zerosOfNums1, zerosOfNums2 = nums1.count(0), nums2.count(0)
    sum1, sum2 = sum(nums1), sum(nums2)
    if zerosOfNums1==0 and zerosOfNums2==0 and sum1 != sum2: return -1
    if zerosOfNums1==0 and zerosOfNums2!=0 and sum1 < sum2 + zerosOfNums2: return -1
    if zerosOfNums1!=0 and zerosOfNums2==0 and sum1 + zerosOfNums1 > sum2: return -1
    return max(sum1+zerosOfNums1,sum2+zerosOfNums2)

nums1=[3,2,0,1,0]
nums2=[6,5,0]
print(minSum(nums1,nums2))