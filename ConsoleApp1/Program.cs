namespace ConsoleApp1
{
    /// <summary>
    /// Main Program 2024-2025 leetcode grind... 
    /// </summary>
    public class Solution
    {
        public static void Main()
        {
            foreach(var val in Merge(new int[6] {1,2,3,0,0,0}, 3, new int[3] {2,5,6}, 3))
            {
                Console.WriteLine(val);
            }
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