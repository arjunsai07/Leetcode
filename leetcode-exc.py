
import numpy as np
from collections import defaultdict
from collections import Counter
class Solution(object):
    def longestConsecutive(self, nums):
        d={}
        s=set(nums)
        l=0
        max_len=0
        if len(nums)==1:
            return 1
        for i,v in enumerate(s):
            d[i]=v 
        for v in d.values():
            if len(s)==1: 
                max_len=1
            elif (v-1) in d.values():
                continue        #if this condition is true, it means v is not start of sequence                
            else:
                curr=v
                l=1
                while curr+1 in s:
                    curr+=1
                    l+=1
                    max_len=max(l,max_len)
        return max_len
    def removeElement(self, nums, val):
        """
        :type nums: List[int]
        :type val: int
        :rtype: int
        """
        for i in range(len(nums) - 1, -1, -1):
            if nums[i] == val:  
                del nums[i] 
        return nums
    def searchInsert(self, nums, target):
        """
        :type nums: List[int]
        :type target: int
        :rtype: int
        """
        low=0
        high=len(nums)-1
        while low<=high:
            mid=(low+high)//2
            if nums[mid]==target:
                return mid
            elif target>nums[mid]:
                low=mid+1
            else:
                high=mid-1
        return low
    def lengthOfLastWord(self, s):
        """
        :type s: str
        :rtype: int
        """
        words=s.split()
        return len(words[-1])
    def addBinary(self, a, b):
        """
        :type a: str
        :type b: str
        :rtype: str
        """
        res=bin(int(a,2)+int(b,2))[2:]
        return res
    def wordPattern(self, pattern, s):
        """
        :type pattern: str
        :type s: str
        :rtype: bool
        """
        dic=defaultdict(list)
        if len(pattern)!=len(s.split()):
            return False
        for letter,word in zip(pattern,s.split()):
            if any(word==v for v in dic.values()):
                continue
            else:
                dic[letter].append(word)
        for k,v in dic.items():
            if(len(set(v))==1):
                continue
            else:
                return False
        return True
    def plusOne(self, digits):
        """
        :type digits: List[int]
        :rtype: List[int]
        """
        n=0
        rev=[]
        for i in digits:
            n=n*10+i 
        n+=1
        while n>0:
            dig=n%10
            rev.append(dig)
            n=n//10
        rev.reverse()
        return rev
    def singleNumber(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        dict={}
        for n in nums:
            if n in dict:
                dict[n]=dict[n]+1
            else:
                dict[n]=1
        for k in dict.keys():
            if dict[k]==1:
                return k
    def majorityElement(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        dict={}
        max,i_max=0,0
        for n in nums:
            if n in dict:
                dict[n]=dict[n]+1
            else:
                dict[n]=1
        for k in dict.keys():
            if dict[k]>max:
                max=dict[k]
                i_max=k
        return i_max    
    def merge(self, nums1, m, nums2, n):
        """
        :type nums1: List[int]
        :type m: int
        :type nums2: List[int]
        :type n: int
        :rtype: None Do not return anything, modify nums1 in-place instead.
        """
    def moveZeroes(self, nums):
        """
        :type nums: List[int]
        :rtype: None Do not return anything, modify nums in-place instead.
        """
        for i in nums:
            if i==0:
                nums.remove(i)
                nums.append(i)
        return nums
    def intersect(self, nums1, nums2):
        """
        :type nums1: List[int]
        :type nums2: List[int]
        :rtype: List[int]
        """
        c1=Counter(nums1)
        c2=Counter(nums2)
        return list((c1&c2).elements())
    def missingNumber(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        d={}
        l=len(nums)
        exp_sum=l*(l+1)//2
        act_sum=sum(nums)
        return exp_sum-act_sum
    def generate(self, numRows):
        """
        :type numRows: int
        :rtype: List[List[int]]
        """
        res=[[1],[1,1]]
    def containsNearbyDuplicate(self, nums, k):
        """
        :type nums: List[int]
        :type k: int
        :rtype: bool
        """
        l=len(nums)
        flag=0
        for i in range(l):
            for j in range(i+1,l):
                if nums[i]==nums[j]: 
                    print(abs(i-j))
                    if abs(i-j)<=k:
                        flag+=1
                    else:
                        flag=0
        return flag>=1 
    def merge(self, nums1, m, nums2, n):
        """
        :type nums1: List[int]
        :type m: int
        :type nums2: List[int]
        :type n: int
        :rtype: None Do not return anything, modify nums1 in-place instead.
        """
        nums1=[x for x in nums1 if x != 0]
        nums1+=nums2
        nums1=sorted(nums1)
    def romanToInt(self, s):
        """
        :type s: str
        :rtype: int
        """
        sum=0
        dic={"I":1,"V":5,"X":10,"L":50,"C":100,"D":500,"M":1000}
        for k in range(0,len(s)):
            if k==len(s)-1:
                sum+=dic[s[k]]
            elif dic[s[k]]<dic[s[k+1]]:
                sum-=dic[s[k]]
            else:
                sum+=dic[s[k]]
        return sum
    def longestCommonPrefix(self, strs):
        """
        :type strs: List[str]
        :rtype: str
        """
        res=[]
        strs=sorted(strs)
        for i in range(len(strs[0])):
            if strs[0][i]==strs[-1][i]:
                res.append(strs[0][i])
            else:
                res=""
                break
        return res
        
      
x=Solution()
print(x.longestCommonPrefix(["flower","flow","flight"]))

#print(x.romanToInt("III"))
#print(x.containsNearbyDuplicate([1,2,3,1,2,3],k=3))
#print(x.intersect([1,2,2,1],[2,2]))
#print(x.moveZeroes([0]))
#print(x.missingNumber([3,0,1]))
#print(x.searchInsert([1,3],2))
#print(x.addBinary("11","1")) 
#print(x.wordPattern("abba","dog dog dog dog"))
#print(x.plusOne([4,3,2,1]))
#print(x.singleNumber([4,1,2,1,2]))
#print(x.majorityElement([3,2,3]))
#print(x.merge([1,2,3,0,0,0],0,[2,5,6],1))