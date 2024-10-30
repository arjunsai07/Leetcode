
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
                break
        res=''.join(res)
        return res
    def strStr(self, haystack, needle):
        """
        :type haystack: str
        :type needle: str
        :rtype: int
        """
        index=haystack.find(needle)
        return index
    def isPalindrome(self, s):
        """
        :type s: str
        :rtype: bool
        """
        s=s.lower().replace(" ", "")
        new=[]
        for i in s:
            if i.isalnum():
                new.append(i)
        return new==new[::-1]
    def validPalindrome(self, s):
        """
        :type s: str
        :rtype: bool
        """
        l=0
        r=len(s)-1
        if s==s[::-1]:
            return True
        while l<r:
            if s[l]!=s[r]:
                int_l=s[:l]+s[l+1:]
                int_r=s[:r]+s[r+1:]
                if int_l==int_l[::-1] or int_r==int_r[::-1]:
                    return True
            i+=1
            k-=1
        return False
    def minimumDifference(self, nums, k):
        """
        :type nums: List[int]
        :type k: int
        :rtype: int
        """
        l,r=0,k-1
        nums=sorted(nums)
        res=max(nums)
        while r<len(nums):
            res=min(res,nums[r]-nums[l])
            l=l+1
            r=r+1
        return res
    def mergeAlternately(self, word1, word2):
        """
        :type word1: str
        :type word2: str
        :rtype: str
        """
        i,j=0,0
        out=[]
        while i<len(word1) and j<len(word2):
            out.append(word1[i])
            out.append(word2[j])
            i+=1
            j+=1
        out.append(word1[i:])
        out.append(word2[j:])
        return "".join(out)
    def reverseString(self, s):
        i=0
        k=len(s)-1
        while i<=k:
            s[i],s[k]=s[k],s[i]
            i+=1
            k-=1
        return s
    def merge(self, nums1, m, nums2, n):
        """
        :type nums1: List[int]
        :type m: int
        :type nums2: List[int]
        :type n: int
        :rtype: None Do not return anything, modify nums1 in-place instead.
        """
        i,j=m,0
        while i<len(nums1) and j<len(nums2):
            nums1[i]=nums2[j]
            i+=1
            j+=1
        nums1.sort()
        return nums1
    def removeDuplicates(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        d={}
        i=0
        while i<len(nums):
            if nums[i] not in d:
                d[nums[i]]=1
                i+=1
            else:
                del nums[i]
        return len(nums)
    def findContentChildren(self, g, s):
        """
        :type g: List[int]
        :type s: List[int]
        :rtype: int
        """
        i=j=0
        while(i<len(g)):
            while(j<len(s) and s[j]<g[i]): #until the size of cookie is less than greed
                j+=1 #the cookie is taken
            if j==len(s): #if it goes out of bounds of s array, need to break
                break
            i+=1
            j+=1
        return i
    def firstPalindrome(self, words):
        """
        :type words: List[str]
        :rtype: str
        """
        flag=0
        for w in words:
            if w==w[::-1]:
                return w
            else:
                continue
        return ""
    def sortArrayByParity(self, nums):
        """
        :type nums: List[int]
        :rtype: List[int]
        """
        i=0
        fix=0
        for i in range(len(nums)):
            if nums[i]%2==0:
                nums[i],nums[fix]=nums[fix],nums[i]
                fix+=1
        return nums
    def reverseWords(self, s):
        """
        :type s: str
        :rtype: str
        """
        out=[]
        for i in s.split(" "):
            out.append(i[::-1])
        return " ".join(out)
    def convert(self, s, numRows):
        """
        :type s: str
        :type numRows: int
        :rtype: str
        """
        L=['']*numRows
        index,step=0,1
        for x in s:
            L[index]+=x
            if index==0:
                step=1 #move down
            elif index==numRows-1:
                step=-1 #move up
            index+=step
        return ''.join(L)
    '''
    def backspaceCompare(self, s, t): #my code
        """
        :type s: str
        :type t: str
        :rtype: bool
        """
        i,j=0,0
        l_s=len(s)
        l_t=len(t)
        while i<l_s:
            if i!=0 and s[i]=='#':
                s=s[:i-1]+s[i+1:]
                l_s-=2
                i=0
                continue
            elif s[0]=='#':
                i+=1
                continue
            i+=1
        while j<l_t:
            if j!=0 and t[j]=='#':
                t=t[:j-1]+t[j+1:]
                l_t-=2
                j=0
                continue
            elif t[0]=='#':
                j+=1
                continue 
            j+=1
        if s==t:
            return True
        else:
            return False
    '''
    def backspaceCompare(self, s, t): #dominic's code
        """
        :type s: str
        :type t: str
        :rtype: bool
        """
        s_lst,t_lst=list(),list()
        for i in s:
            if i=='#':
                if len(s_lst) is not 0:
                    s_lst.pop()
            else:
                    s_lst.append(i)
        for j in t:
            if j=='#':
                if len(t_lst) is not 0:
                    t_lst.pop()
            else:
                    t_lst.append(j)
        return s_lst==t_lst
    def arrayStringsAreEqual(self, word1, word2):
        a1,a2="",""
        for i in word1:
            a1+=""+i
        for j in word2:
            a2+=""+j
        return a1==a2
    def removeDuplicates2(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        d={}
        i=0
        while i<len(nums):
            if nums[i] not in d:
                d[nums[i]]=1
                i+=1
            elif nums[i] in d and d[nums[i]]<2:
                d[nums[i]]+=1
                i+=1
            else:
                del nums[i]
        return len(nums)
    def twoSum2(self, numbers, target): #copied
        """
        :type numbers: List[int]
        :type target: int
        :rtype: List[int]
        """
        l,r=0,len(numbers)-1
        while l<r:
            sum=numbers[l]+numbers[r]
            if sum>target: #sum is bigger than target
                r-=1
            elif sum<target: #sum is lesser than target
                l+=1
            else: #sum is equal to target
                return [l+1,r+1]
    def threeSum(self, nums): #twosum2 is basic for this problem
        """
        :type nums: List[int]
        :rtype: List[List[int]]
        """
        nums=sorted(nums)
        i=0
        out=[]
        while i<len(nums):
            l,r=i+1,len(nums)-1
            req_sum=0-nums[i]
            while l<r:
                sum=nums[l]+nums[r]
                if sum>req_sum: #sum is bigger than target
                    r-=1
                elif sum<req_sum: #sum is lesser than target
                    l+=1
                else: #sum is equal to target
                    if [nums[i],nums[l],nums[r]] not in out:
                        out.append([nums[i],nums[l],nums[r]])
                    r-=1
            i+=1
        return out
    def numSubseq(self, nums, target):
        l,r=0,len(nums)-1
        count,mod=0,(10**9+7)
        nums.sort()
        while l<=r:
            if nums[l]+nums[r]<=target:
                count+=(2**(r-l))%mod
                l+=1
            else:
                r-=1
        return count%mod
    def rotate(self, nums, k): #copied, more than 1 hour
        k=k%len(nums)
        nums.reverse()
        nums[:k]=reversed(nums[:k])
        nums[k:]=reversed(nums[k:])
        return nums
    def rearrangeArray(self, nums): #neighbor averages wont be equal to middle number if neighbors both are either larger than middle number or are both smaller than middle number
        """
        :type nums: List[int]
        :rtype: List[int]
        """
        i,j,k=0,len(nums)-1,0
        nums.sort()
        inter=[0]*len(nums)
        while i<=j and k<=j:
            inter[k]=nums[i]
            i,k=i+1,k+2
        k=1
        while i<=j and k<=j:
            inter[k]=nums[i]
            i,k=i+1,k+2
        return inter
    def numRescueBoats(self, people, limit):
        """
        :type people: List[int]
        :type limit: int
        :rtype: int
        """ 
        people.sort()
        l,r,res,sum=0,len(people)-1,0,0
        while l<=r:
            if l==r:
                sum=people[l]
            else:
                sum=people[l]+people[r]
            if sum>limit:
                res+=1 
                r-=1
            else:
                res+=1
                l+=1
                r-=1
        return res
                
x=Solution()  
print(x.numRescueBoats([3,2,2,1],3))
#print(x.rearrangeArray([6,2,0,9,7]))
#print(x.rotate([1,2,3,4,5,6,7],3))
#print(x.backspaceCompare())  
#print(x.numSubseq([3,5,6,7],9))
#print(x.threeSum([-2,0,1,1,2])) 
#print(x.twoSum2([-1,0],-1))
#print(x.removeDuplicates2([0,0,1,1,1,1,2,3,3]))
#print(x.arrayStringsAreEqual(["abc", "d", "defg"],["abcddefg"]))
#print(x.backspaceCompare("y#fo##f","y#f#o##f"))
#print(x.convert("PAYPALISHIRING", 3))
#print(x.reverseWords("Let's take LeetCode contest"))
#print(x.sortArrayByParity([3,1,2,4]))
#print(x.firstPalindrome(["def","ghi"]))
#print(x.findContentChildren([1,2],[1,2,3]))
#print(x.removeDuplicates([0,0,1,1,1,2,2,3,3,4]))
#print(x.merge([1,2,3,0,0,0],3,[2,5,6],3))
#print(x.reverseString(["h","e","l","l","o"]))
#print(x.mergeAlternately("abc","pqr"))
#print(x.minimumDifference([9,4,1,7],3))
#print(x.validPalindrome(p))
#print(x.isPalindrome(" "))
#print(x.longestCommonPrefix(["flower","flow","flight"]))
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
