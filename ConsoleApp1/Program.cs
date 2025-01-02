namespace ConsoleApp1
{
    /// <summary>
    /// Main Program 2024-2025 leetcode grind... 
    /// </summary>
    public class Solution
    {
        public static void Main()
        {
            foreach (var digit in PlusOne(new int[] { 1, 2, 3 }))
            {
                Console.WriteLine(digit);
            }
        }

        /// <summary>
        /// You are given a large integer represented as an integer array digits,
        /// where each digits[i] is the ith digit of the integer. 
        /// The digits are ordered from most significant to least significant in left-to-right order. 
        /// The large integer does not contain any leading 0's.
        /// </summary>
        /// <param name="digits"></param>
        /// <returns></returns>
        public static int[] PlusOne(int[] digits)
        {
            if (digits.Length == 0) Array.Empty<int>();

            int n = digits.Length;
            for (int i = n - 1; i >= 0; i--)
            {
                if (digits[i] < 9)
                {
                    digits[i]++;
                    return digits;
                }
                digits[i] = 0;
            }
            int[] newNumber = new int[n + 1];
            newNumber[0] = 1;

            return newNumber;
        }

        /// <summary>
        /// Given an array nums of size n, return the majority element.
        /// </summary>
        /// <param name="nums"></param>
        /// <returns></returns>
        public static int MajorityElement(int[] nums)
        {
            if (nums.Length == 0) return 0;

            int candidate = 0, count = 0;

            // Phase 1: Find a Candidate
            foreach (int num in nums)
            {
                if (count == 0)
                {
                    candidate = num; // Set a new candidate
                    count = 1;       // Reset the count
                }
                else if (num == candidate)
                {
                    count++; // Increment count for the same candidate
                }
                else
                {
                    count--; // Decrement count for a different element
                }
            }

            // Phase 2: Verify the Candidate
            count = 0;
            foreach (int num in nums)
            {
                if (num == candidate) count++;
            }

            return count > nums.Length / 2 ? candidate : 0;
        }

        /// <summary>
        /// Given an integer array nums and an integer val, remove all occurrences of val in nums in-place. 
        /// The order of the elements may be changed. Then return the number of elements in nums which
        /// are not equal to val.
        /// </summary>
        /// <param name="nums"></param>
        /// <param name="val"></param>
        /// <returns></returns>
        public static int RemoveElement(int[] nums, int val)
        {
            if (nums.Length == 0) return 0;
            if (val == 0) return 0;

            int writeIndex = 0;
            for (int i = 0; i < nums.Length; i++)
            {
                if (nums[i] != val)
                {
                    nums[writeIndex] = nums[i];
                    writeIndex++;
                }
            }
            return writeIndex;
        }

        /// <summary>
        /// Given two strings needle and haystack, 
        /// return the index of the first occurrence of needle in haystack, 
        /// or -1 if needle is not part of haystack.
        /// </summary>
        /// <param name="haystack"></param>
        /// <param name="needle"></param>
        /// <returns></returns>
        public static int StrStr(string haystack, string needle)
        {
            if (string.IsNullOrEmpty(haystack)) return -1;
            if (string.IsNullOrEmpty(needle)) return -1;

            int result = -1;
            int matchLoc = 0;

            for (int i = 0; i < haystack.Length; i++)
            {
                if (haystack[i] == needle[matchLoc])
                {
                    matchLoc++;
                    if (matchLoc == needle.Length)
                    {
                        result = i - matchLoc + 1;
                        break;
                    }
                }
                else
                {
                    i -= matchLoc;
                    matchLoc = 0;
                }
            }
            return result;
        }

        /// <summary>
        /// Given a string s containing just the characters 
        /// '(', ')', '{', '}', '[' and ']', determine if the input string is valid.
        /// </summary>
        /// <param name="s"></param>
        /// <returns></returns>
        public static bool IsValid(string s)
        {
            if (string.IsNullOrEmpty(s)) return false;

            Stack<char> stack = new Stack<char>();

            foreach (char c in s)
            {
                if (c == '(' || c == '{' || c == '[')
                {
                    stack.Push(c);
                }
                else if (c == ')' && stack.Count > 0 && stack.Peek() == '(')
                {
                    stack.Pop();
                }
                else if (c == '}' && stack.Count > 0 && stack.Peek() == '{')
                {
                    stack.Pop();
                }
                else if (c == ']' && stack.Count > 0 && stack.Peek() == '[')
                {
                    stack.Pop();
                }
                else
                {
                    return false;
                }
            }

            return true;
        }

        /// <summary>  
        /// Find the longest common prefix in an array of strings.  
        /// </summary>  
        /// <param name="s"></param>  
        /// <returns></returns>  
        public static int LengthOfLastWord(string s)
        {
            if (string.IsNullOrEmpty(s)) return s.Length;

            int n = s.Length;
            int result = 0;
            for (int i = n - 1; i >= 0; i--)
            {
                if (s[i] != ' ')
                {
                    result++;
                }
                else if (result > 0)
                {
                    return result;
                }
            }
            return result;
        }

        /// <summary>
        /// Merge two sorted arrays into one sorted array.
        /// </summary>
        /// <param name="nums1"></param>
        /// <param name="m"></param>
        /// <param name="nums2"></param>
        /// <param name="n"></param>
        /// <returns></returns>
        public static int[] Merge(int[] nums1, int m, int[] nums2, int n)
        {
            int i = m - 1; // Pointer for nums1
            int j = n - 1; // Pointer for nums2
            int k = m + n - 1; // Pointer for the end of nums1. -1 because zero index based.

            // Merge from the back
            while (j >= 0)
            {
                if (i >= 0 && nums1[i] > nums2[j])
                {
                    nums1[k] = nums1[i];
                    i--;
                }
                else
                {
                    nums1[k] = nums2[j];
                    j--;
                }
                k--;
            }

            return nums1;
        }

        /// <summary>
        /// Check if a number is a palindrome.
        /// </summary>
        /// <param name="x"></param>
        /// <returns></returns>
        public static bool IsPalindrome(int x)
        {
            int r = 0;
            int c = x;

            while (c > 0)
            {
                r = r * 10 + c % 10;
                c /= 10;
            }
            return r == x;
        }

        /// <summary>
        /// Find the two indices that sum up to the target.
        /// </summary>
        /// <param name="numbers"></param>
        /// <param name="target"></param>
        /// <returns></returns>
        public static int[] TwoSum(int[] numbers, int target)
        {
            if (numbers.Length == 0) return new int[0];
            // Initialize two pointers, one at the start (left) and one at the end (right) of the array.

            // Check the sum of the elements at the two pointers.

            // If the sum equals the target, return the indices.

            // If the sum is less than the target, move the left pointer to the right.

            // If the sum is greater than the target, move the right pointer to the left.
            var left = 0;
            var right = numbers.Length - 1;

            // Sorted by ascending order
            while (left < right)
            {
                var total = numbers[left] + numbers[right];
                if (total == target)
                {
                    return new int[2] { left + 1, right + 1 };
                }
                else if (total > target)
                {
                    right -= 1;
                }
                else
                {
                    left += 1;
                }
            }

            return new int[0];
        }
    }
}