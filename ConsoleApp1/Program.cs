using System.ComponentModel;
using System.Data.Common;
using System.Data.SqlTypes;
using System.Diagnostics.Metrics;
using System.IO;
using System.Security.Cryptography;
using System.Text;
using static System.Runtime.InteropServices.JavaScript.JSType;

namespace ConsoleApp1
{
    public class ListNode
    {
        public int val;
        public ListNode next;
        public ListNode(int val = 0, ListNode next = null)
        {
            this.val = val;
            this.next = next;
        }
    }

    /// <summary>
    /// Main Program 2024-2025 leetcode grind... 
    /// </summary>
    public class Solution
    {
        public static void Main()
        {
            Console.WriteLine(LongestConsecutive([0, 3, 7, 2, 5, 8, 4, 6, 0, 1]));
        }

        public class MinStack
        {
            Stack<(int val, int minVal)> stack;
            int minVal = int.MaxValue;
            public MinStack()
            {
                stack = new Stack<(int, int)>();
            }
            public void Push(int val)
            {
                if (minVal > val)
                {
                    minVal = val;
                }
                stack.Push((val, minVal));
            }
            public void Pop()
            {
                stack.Pop();
                if (stack.Count > 0)
                {
                    minVal = stack.Peek().minVal;
                }
                else
                {
                    minVal = int.MaxValue;
                }
            }
            public int Top()
            {
                return stack.Peek().val;
            }
            public int GetMin()
            {
                return stack.Peek().minVal;
            }
        }

        /// <summary>
        /// You are given an absolute path for a Unix-style file system, 
        /// which always begins with a slash '/'. Your task is to transform 
        /// this absolute path into its simplified canonical path.
        /// </summary>
        /// <param name="path"></param>
        /// <returns></returns>
        public static string SimplifyPath(string path)
        {
            if (string.IsNullOrEmpty(path)) return string.Empty;
            var finalizedPath = string.Empty;

            var stack = new Stack<string>();
            var parts = path.Split('/');
            foreach (var part in parts)
            {
                if (part == "..")
                {
                    if (stack.Count > 0)
                    {
                        stack.Pop();
                    }
                }
                else if (part != "." && part != "")
                {
                    stack.Push(part);
                }
            }
            var result = new StringBuilder();
            while (stack.Count > 0)
            {
                result.Insert(0, stack.Pop());
                result.Insert(0, "/");
            }
            return result.Length == 0 ? "/" : result.ToString();
        }

        /// <summary>
        /// There are some spherical balloons taped onto a flat wall that represents the XY-plane. 
        /// The balloons are represented as a 2D integer array points where points[i] = [xstart, xend] 
        /// denotes a balloon whose horizontal diameter stretches between xstart and xend. 
        /// You do not know the exact y-coordinates of the balloons.
        /// Arrows can be shot up directly vertically(in the positive y-direction) from different
        /// points along the x-axis.A balloon with xstart and xend is burst by an arrow shot at x if
        /// xstart <= x <= xend.There is no limit to the number of arrows that can be shot.
        /// A shot arrow keeps traveling up infinitely, bursting any balloons in its path.
        /// Given the array points, return the minimum number of arrows that must be shot to burst all balloons.
        /// </summary>
        /// <param name="points"></param>
        /// <returns></returns>
        public static int FindMinArrowShots(int[][] points)
        {
            if (points.Length == 0) return 0;

            // Sort the balloons based on their end coordinates
            Array.Sort(points, (a, b) => a[1].CompareTo(b[1]));

            int arrows = 1;
            int prevEnd = points[0][1];

            // Count the number of non-overlapping intervals
            for (int i = 1; i < points.Length; ++i)
            {
                if (points[i][0] > prevEnd)
                {
                    arrows++;
                    prevEnd = points[i][1];
                }
            }

            return arrows;
        }

        /// <summary>
        /// You are given an array of non-overlapping intervals intervals where intervals[i] = [starti, endi] 
        /// represent the start and the end of the ith interval and intervals is sorted in ascending order by starti. 
        /// You are also given an interval newInterval = [start, end] that represents the start and end of another interval.
        /// 
        /// Insert newInterval into intervals such that intervals is still sorted in ascending order by starti and intervals
        /// still does not have any overlapping intervals (merge overlapping intervals if necessary).
        /// Return intervals after the insertion.
        /// </summary>
        /// <param name="intervals"></param>
        /// <param name="newInterval"></param>
        /// <returns></returns>
        public static int[][] Insert(int[][] intervals, int[] newInterval)
        {
            if (intervals.Length == 0) return [];

            var result = new List<int[]>();

            // Iterate through intervals and add non-overlapping intervals before newInterval
            int i = 0;
            while (i < intervals.Length && intervals[i][1] < newInterval[0])
            {
                result.Add(intervals[i]);
                i++;
            }

            // Merge overlapping intervals
            while (i < intervals.Length && intervals[i][0] <= newInterval[1])
            {
                newInterval[0] = Math.Min(newInterval[0], intervals[i][0]);
                newInterval[1] = Math.Max(newInterval[1], intervals[i][1]);
                i++;
            }

            // Add merged newInterval
            result.Add(newInterval);

            // Add non-overlapping intervals after newInterval
            while (i < intervals.Length)
            {
                result.Add(intervals[i]);
                i++;
            }

            return result.ToArray();
        }

        /// <summary>
        /// Given an array of intervals where intervals[i] = [starti, endi], merge all overlapping intervals,
        /// and return an array of the non-overlapping intervals that cover all the intervals in the input.
        /// </summary>
        /// <param name="intervals"></param>
        /// <returns></returns>
        public int[][] Merge(int[][] intervals)
        {
            if (intervals.Length == 0) return [];

            var result = new List<int[]>();
            Array.Sort(intervals, (a, b) => a[0] - b[0]);
            result.Add(intervals[0]);
            for (int i = 1; i < intervals.Length; i++)
            {
                if (intervals[i][0] <= result[result.Count - 1][1])
                {
                    result[result.Count - 1][1] = Math.Max(result[result.Count - 1][1], intervals[i][1]);
                }
                else
                {
                    result.Add(intervals[i]);
                }
            }
            return result.ToArray();
        }

        /// <summary>
        /// Given an unsorted array of integers nums, 
        /// return the length of the longest consecutive elements sequence.
        /// You must write an algorithm that runs in O(n) time.
        /// </summary>
        /// <param name="nums"></param>
        /// <returns></returns>
        public static int LongestConsecutive(int[] nums)
        {
            if (nums.Length == 0) return 0;

            HashSet<int> set = new HashSet<int>(nums);
            int maxLength = 0;

            foreach (int num in nums)
            {
                if (set.Contains(num - 1)) continue;

                int length = 0;
                while (set.Contains(num + length)) length++;

                maxLength = Math.Max(maxLength, length);
            }

            return maxLength;
        }

        public IList<IList<string>> GroupAnagrams(string[] strs)
        {
            if (strs.Length == 0) return new List<IList<string>>();
            var map = new Dictionary<string, List<string>>();

            foreach (var str in strs)
            {
                var key = String.Concat(str.OrderBy(c => c));
                if (!map.ContainsKey(key)) map[key] = new List<string>();
                map[key].Add(str);
            }

            return map.Values.Cast<IList<string>>().ToList();
        }

        /// <summary>
        /// Determine if a 9 x 9 Sudoku board is valid. Only the filled cells need to be validated according 
        /// to the following rules: Each row must contain the digits 1-9 without repetition. 
        /// Each column must contain the digits 1-9 without repetition.
        /// Each of the nine 3 x 3 sub-boxes of the grid must contain the digits 1-9 without repetition.
        /// </summary>
        /// <param name="board"></param>
        /// <returns></returns>
        public static bool IsValidSudoku(char[][] board)
        {
            if (!board.Any()) return false;

            HashSet<char>[] row = new HashSet<char>[9];
            HashSet<char>[] col = new HashSet<char>[9];
            HashSet<char>[] box = new HashSet<char>[9];
            for (int i = 0; i < 9; i++)
            {
                row[i] = new HashSet<char>();
                col[i] = new HashSet<char>();
                box[i] = new HashSet<char>();
            }

            for (int r = 0; r < board.Length; r++)
            {
                for (int c = 0; c < board[r].Length; c++)
                {
                    char elem = board[r][c];
                    if (elem == '.')
                    {
                        continue;
                    }

                    if (!row[r].Add(elem))
                    {
                        return false;
                    }

                    if (!col[c].Add(elem))
                    {
                        return false;
                    }

                    int b = (3 * (r / 3)) + (c / 3);
                    if (!box[b].Add(elem))
                    {
                        return false;
                    }
                }
            }

            return true;
        }

        /// <summary>
        /// Given a string s, find the length of the longest
        /// substring without repeating characters.
        /// </summary>
        /// <param name="s"></param>
        /// <returns></returns>
        public static int LengthOfLongestSubstring(string s)
        {
            if (string.IsNullOrEmpty(s)) return 0;

            var charSet = new HashSet<char>();
            int left = 0, right = 0, maxLength = 0;
            while (right < s.Length)
            {
                if (!charSet.Contains(s[right]))
                {
                    charSet.Add(s[right]);
                    right++;
                    maxLength = Math.Max(maxLength, charSet.Count);
                }
                else
                {
                    charSet.Remove(s[left]);
                    left++;
                }
            }
            return maxLength;
        }

        /// <summary>
        /// Given an array of positive integers nums and a positive integer target, 
        /// return the minimal length of a subarray whose sum is greater than or 
        /// equal to target.If there is no such subarray, return 0 instead.
        /// </summary>
        /// <param name="target"></param>
        /// <param name="nums"></param>
        /// <returns></returns>
        public static int MinSubArrayLen(int target, int[] nums)
        {
            if (target == 0) return 0;
            if (nums.Length == 0) return 0;

            int left = 0, right = 0, sum = 0, min = Int32.MaxValue;
            while (right < nums.Length)
            {
                sum += nums[right];
                right++;
                while (sum >= target)
                {
                    min = Math.Min(min, right - left);
                    sum -= nums[left];
                    left++;
                }
            }
            return min == Int32.MaxValue ? 0 : min;
        }

        public static IList<IList<int>> ThreeSum(int[] nums)
        {
            if (nums.Length == 0) return new List<IList<int>>();

            Array.Sort(nums);
            IList<IList<int>> result = new List<IList<int>>();

            for (int i = 0; i < nums.Length; i++)
            {
                if (i > 0 && nums[i] == nums[i - 1])
                {
                    continue; // Skip duplicate values for i
                }
                int j = i + 1, k = nums.Length - 1;
                while (j < k)
                {
                    int sum = nums[i] + nums[j] + nums[k];
                    if (sum == 0)
                    {
                        result.Add(new List<int> { nums[i], nums[j], nums[k] });
                        j++;
                        k--;
                        while (j < k && nums[j] == nums[j - 1]) j++;
                        while (j < k && nums[k] == nums[k + 1]) k--;
                    }
                    else if (sum < 0)
                    {
                        j++;
                    }
                    else
                    {
                        k--;
                    }
                }
            }

            return result;
        }

        /// <summary>
        /// You are given an integer array height of length n. There are n vertical lines
        /// drawn such that the two endpoints of the ith line are (i, 0) and (i, height[i]). 
        /// Find two lines that together with the x-axis form a container, such that the 
        /// container contains the most water. Return the maximum amount of water a container can store.
        /// Notice that you may not slant the container.
        /// </summary>
        /// <param name="height"></param>
        public static int MaxArea(int[] height)
        {
            if (height.Length == 0) return 0;

            int maxArea = 0;
            int i = 0, j = height.Length - 1;

            while (i < j)
            {
                int curWidth = j - i;
                int curHeight = Math.Min(height[i], height[j]);
                maxArea = Math.Max(maxArea, curWidth * curHeight);

                if (height[i] <= height[j])
                {
                    i++;
                }
                else
                {
                    j--;
                }
            }
            return maxArea;
        }

        /// <summary>
        /// Given a 1-indexed array of integers numbers that is already sorted in non-decreasing order, 
        /// find two numbers such that they add up to a specific target number. Let these two numbers 
        /// be numbers[index1] and numbers[index2] where 1 <= index1 < index2 <= numbers.length.
        /// Return the indices of the two numbers, index1 and index2, added by one as an integer array[index1, index2] of length 2.
        /// The tests are generated such that there is exactly one solution.You may not use the same element twice.
        /// Your solution must use only constant extra space.
        /// </summary>
        /// <param name="numbers"></param>
        /// <param name="target"></param>
        /// <returns></returns>
        public static int[] TwoSumTwo(int[] numbers, int target)
        {
            var intArray = new int[0];
            if (target == 0) return intArray;
            if (numbers.Length == 0) return intArray;

            int left = 0;
            int right = numbers.Length - 1;
            while (left < right)
            {
                int sum = numbers[left] + numbers[right];
                if (sum == target) break;
                if (sum < target) left++;
                if (sum > target) right--;
            }
            return new int[] { left + 1, right + 1 };
        }

        /// <summary>
        /// The string "PAYPALISHIRING" is written in a zigzag pattern on a given number of rows like this:
        /// (you may want to display this pattern in a fixed font for better legibility)
        /// </summary>
        /// <param name="s"></param>
        /// <param name="numRows"></param>
        /// <returns></returns>
        public static string Convert(string s, int numRows)
        {
            if (string.IsNullOrEmpty(s) || numRows == 0) return string.Empty;
            
            if (numRows == 1)
            {
                return s;
            }

            Span<char> result = stackalloc char[s.Length];

            var resultIndex = 0;
            var period = numRows * 2 - 2;

            for (int row = 0; row < numRows; row++)
            {
                var increment = 2 * row;

                for (int i = row; i < s.Length; i += increment)
                {
                    result[resultIndex++] = s[i];

                    if (increment != period)
                    {
                        increment = period - increment;
                    }
                }
            }

            return result.ToString();
        }

        /// <summary>
        /// Given an input string s, reverse the order of the words.
        /// A word is defined as a sequence of non-space characters.The words in s will be separated by at least one space.
        /// Return a string of the words in reverse order concatenated by a single space.
        /// Note that s may contain leading or trailing spaces or multiple spaces between two words.
        /// The returned string should only have a single space separating the words. Do not include any extra spaces.
        /// </summary>
        /// <param name="s"></param>
        /// <returns></returns>
        public static string ReverseWords(string s)
        {
            if (string.IsNullOrEmpty(s)) return string.Empty;
            var reverse = new StringBuilder();

            var words = s.Trim().Split(' ');
            foreach (var word in words)
            {
                if (string.IsNullOrEmpty(word))
                    continue;
                reverse = new StringBuilder(word + " " + reverse.ToString());
            }
            return reverse.ToString().Trim();
        }

        /// <summary>
        /// Seven different symbols represent Roman numerals with the following values:
        /// Symbol	Value
        /// I	1
        /// V	5
        /// X	10
        /// L	50
        /// C	100
        /// D	500
        /// M	1000
        /// </summary>
        /// <param name="num"></param>
        /// <returns></returns>
        public static string IntToRoman(int num)
        {
            if (num == 0) return string.Empty;
            var result = new StringBuilder();

            Dictionary<int, string> rdmap = new Dictionary<int, string>()
            {
                {1000, "M"}, {900, "CM"}, {500, "D"}, {400, "CD"},
                {100, "C"}, {90, "XC"}, {50, "L"}, {40, "XL"},
                {10, "X"}, {9, "IX"}, {5, "V"}, {4, "IV"}, {1, "I"}
            };
            
            foreach (var (value, symbol) in rdmap)
            {
                while (num >= value)
                {
                    result.Append(symbol);
                    num -= value;
                }
            }
            return result.ToString();
        }

        /// <summary>
        /// There are n gas stations along a circular route, where the amount of gas at the ith station is gas[i].
        /// You have a car with an unlimited gas tank and it costs cost[i] of gas to travel from the ith station 
        /// to its next(i + 1)th station. You begin the journey with an empty tank at one of the gas stations.
        /// Given two integer arrays gas and cost, return the starting gas station's index if you can travel around 
        /// the circuit once in the clockwise direction, otherwise return -1. If there exists a solution, it is guaranteed to be unique.
        /// </summary>
        /// <param name="gas"></param>
        /// <param name="cost"></param>
        /// <returns></returns>
        public static int CanCompleteCircuit(int[] gas, int[] cost)
        {
            if (gas.Length == 0) return 0;
            if (cost.Length == 0) return 0;

            int sum = 0;
            int maxIndex = -1;
            int maxSum = int.MinValue;

            for (int i = gas.Length - 1; i >= 0; i--)
            {
                sum += gas[i] - cost[i];

                if (sum > maxSum)
                {
                    maxIndex = i;
                    maxSum = sum;
                }
            }

            return sum < 0 ? -1 : maxIndex;
        }

        /// <summary>
        /// Given an integer array nums, return an array answer such that answer[i] is equal 
        /// to the product of all the elements of nums except nums[i].
        /// The product of any prefix or suffix of nums is guaranteed to fit in a 32-bit integer.
        /// You must write an algorithm that runs in O(n) time and without using the division operation.
        /// </summary>
        /// <param name="nums"></param>
        /// <returns></returns>
        public int[] ProductExceptSelf(int[] nums)
        {
            if (nums.Length == 0) return new int[0];

            int n = nums.Length;
            int[] res = new int[n];

            // Initialize the result array to store left products.
            int prod = 1;
            for (int i = 0; i < n; i++)
            {
                res[i] = prod;
                prod *= nums[i];
            }

            // Multiply the result array with the right products.
            prod = 1;
            for (int i = n - 1; i >= 0; i--)
            {
                res[i] *= prod;
                prod *= nums[i];
            }

            return res;
        }

        /// <summary>
        /// Given an array of integers citations where citations[i] is the number of citations a researcher received for their ith paper, 
        /// return the researcher's h-index. According to the definition of h-index on Wikipedia: The h-index is defined as the maximum 
        /// value of h such that the given researcher has published at least h papers that have each been cited at least h times.
        /// </summary>
        /// <param name="citations"></param>
        /// <returns></returns>
        public static int HIndex(int[] citations)
        {
            if (citations.Length == 0) return 0;

            int n = citations.Length;
            int[] count = new int[n + 1];

            foreach (int c in citations)
            {
                if (c >= n)
                {
                    count[n]++;
                }
                else
                {
                    count[c]++;
                }
            }

            int total = 0;
            for (int i = n; i >= 0; i--)
            {
                total += count[i];
                if (total >= i)
                {
                    return i;
                }
            }

            return 0;
        }

        /// <summary>
        /// You are given a 0-indexed array of integers nums of length n. You are initially positioned at nums[0].
        /// Each element nums[i] represents the maximum length of a forward jump from index i.In other words, 
        /// if you are at nums[i], you can jump to any nums[i + j] where:
        /// 0 <= j <= nums[i] and
        /// i + j<n
        /// Return the minimum number of jumps to reach nums[n - 1]. The test cases are generated such that 
        /// you can reach nums[n - 1].
        /// </summary>
        /// <param name="nums"></param>
        /// <returns></returns>
        public static int Jump(int[] nums)
        {
            if (nums.Length == 0) return 0;

            int jumps = 0, farthest = 0, end = 0;

            for (int i = 0; i < nums.Length - 1; i++)
            {
                // Update the farthest point we can reach
                farthest = Math.Max(farthest, i + nums[i]);

                // If we've reached the current end, we need to jump
                if (i == end)
                {
                    jumps++;
                    end = farthest; // Update the range for the next jump
                }
            }

            return jumps;
        }

        /// <summary>
        /// Can jump from end to start.
        /// </summary>
        /// <param name="nums"></param>
        /// <returns></returns>
        public static bool CanJump(int[] nums)
        {
            if (nums.Length == 0) return false;

            int finishIndex = nums.Length - 1;
            for (int i = nums.Length - 1; i >= 0; i--)
            {
                if (i + nums[i] >= finishIndex)
                {
                    if (i == 0) return true;
                    finishIndex = i;
                }
            }
            return false;
        }

        /// <summary>
        /// You are given an integer array prices where prices[i] is the price of a given stock
        /// on the ith day. On each day, you may decide to buy and/or sell the stock. You can 
        /// only hold at most one share of the stock at any time. However, you can buy it then 
        /// immediately sell it on the same day. Find and return the maximum profit you can achieve.
        /// </summary>
        /// <param name="prices"></param>
        /// <returns></returns>
        public static int MaxProfit(int[] prices)
        {
            if (prices.Length == 0) return 0;

            int profit = 0;

            for (int i = 1; i < prices.Length; i++)
            {
                if (prices[i] > prices[i - 1])
                    profit += prices[i] - prices[i - 1];
            }

            return profit;
        }

        /// <summary>
        /// Given an integer array nums, rotate the array to the right by k steps, where k is non-negative.
        /// </summary>
        /// <param name="nums"></param>
        /// <param name="k"></param>
        public static void Rotate(int[] nums, int k)
        {
            if (nums.Length == 0) return;
            if (k == 0) return;

            int n = nums.Length;
            k = k % n;
            int[] res = new int[n];
            for (int i = 0; i < n; i++)
            {
                res[(i + k) % n] = nums[i];
            }
            for (int i = 0; i < n; i++)
            {
                nums[i] = res[i];
            }
        }

        /// <summary>
        /// Given an integer array nums sorted in non-decreasing (increasing) order, remove some duplicates 
        /// in-place such that each unique element appears at most twice. The relative order of 
        /// the elements should be kept the same. Since it is impossible to change the length of the array
        /// in some languages, you must instead have the result be placed in the first part of the array nums. 
        /// More formally, if there are k elements after removing the duplicates, then the first k elements 
        /// of nums should hold the final result. It does not matter what you leave beyond the first k elements.
        /// Return k after placing the final result in the first k slots of nums.
        /// Do not allocate extra space for another array. You must do this by modifying the input array in-place with O(1) extra memory.
        /// </summary>
        /// <param name="nums"></param>
        /// <returns></returns>
        public static int RemoveDuplicates(int[] nums)
        {
            if (nums.Length == 0) return 0;

            var substituteIndex = 0;
            for (var i = 0; i < nums.Length; i++)
            {
                // Skips on the third consecutive time the element was found. Since at most we only need two. 
                if (substituteIndex - 2 >= 0 && nums[substituteIndex - 2] == nums[i])
                {
                    continue;
                }
                nums[substituteIndex] = nums[i];
                substituteIndex++;
            }
            return substituteIndex;
        }

        /// <summary>
        /// You are given the heads of two sorted linked lists list1 and list2.
        /// Merge the two lists into one sorted list.The list should be made by splicing together the nodes of the first two lists.
        /// Return the head of the merged linked list.
        /// </summary>
        /// <param name="list1"></param>
        /// <param name="list2"></param>
        /// <returns></returns>
        public static ListNode MergeTwoLists(ListNode list1, ListNode list2)
        {
            ListNode dummy = new ListNode(0);
            ListNode prev = dummy;

            ListNode p1 = list1, p2 = list2;
            while (p1 != null && p2 != null)
            {
                if (p1.val < p2.val)
                {
                    prev.next = p1;
                    p1 = p1.next;
                }
                else
                {
                    prev.next = p2;
                    p2 = p2.next;
                }
                prev = prev.next;
            }

            prev.next = p1 != null ? p1 : p2;

            return dummy.next;
        }

        /**
        * Definition for singly-linked list.
        * public class ListNode {
        *     public int val;
        *     public ListNode next;
        *     public ListNode(int x) {
        *         val = x;
        *         next = null;
        *     }
        * }
        *   var head = new ListNode(3);
        *   head.next = new ListNode(2);
        *   head.next = new ListNode(0);
        *   head.next = new ListNode(-5);
        *   
        *   Given head, the head of a linked list, determine if the linked list has a cycle in it.
        *   There is a cycle in a linked list if there is some node in the list that can be reached
        *   again by continuously following the next pointer. Internally, pos is used to denote the 
        *   index of the node that tail's next pointer is connected to. Note that pos is not passed as a parameter.
        *   Return true if there is a cycle in the linked list. Otherwise, return false.    
        */
        public static bool HasCycle(ListNode head)
        {
            if (head == null) return false;

            HashSet<ListNode> finder = new HashSet<ListNode>();

            ListNode current = head;

            while (current != null)
            {
                if (finder.Contains(current))
                    return true;

                finder.Add(current);
                current = current.next;
            }

            return false;
        }

        /// <summary>
        /// You are given a sorted unique integer array nums.
        /// A range[a, b] is the set of all integers from a to b(inclusive).
        /// Return the smallest sorted list of ranges that cover all the numbers 
        /// in the array exactly.That is, each element of nums is covered by exactly
        /// one of the ranges, and there is no integer x such that x is in one of the ranges but not in nums.
        /// Each range[a, b] in the list should be output as:
        /// </summary>
        /// <param name="nums"></param>
        /// <returns></returns>
        public static IList<string> SummaryRanges(int[] nums)
        {
            if (nums.Length == 0) return new List<string>();

            var result = new List<string>();
            for (int i = 0; i < nums.Length; i++)
            {
                int start = nums[i];
                while (i < nums.Length - 1 && nums[i + 1] - nums[i] == 1)
                {
                    i++;
                }
                if (start != nums[i])
                    result.Add($"{start}->{nums[i]}");
                else
                    result.Add($"{start}");
            }

            return result;
        }

        /// <summary>
        /// Given an integer array nums and an integer k, 
        /// return true if there are two distinct indices i and j
        /// in the array such that nums[i] == nums[j] and abs(i - j) <= k.
        /// </summary>
        /// <param name="nums"></param>
        /// <param name="k"></param>
        /// <returns></returns>
        public static bool ContainsNearbyDuplicate(int[] nums, int k)
        {
            if (nums.Length == 0) return false;
            if (k == 0) return false;

            var numIndices = new HashSet<int>();

            for (int i = 0; i < nums.Length; i++)
            {
                if (numIndices.Contains(nums[i]))
                {
                    return true;
                }

                numIndices.Add(nums[i]);

                if (numIndices.Count > k)
                {
                    numIndices.Remove(nums[i - k]);
                }
            }

            return false;
        }

        /// <summary>
        /// Given a sorted array of distinct integers and a target value, 
        /// return the index if the target is found. 
        /// If not, return the index where it would be if it were inserted in order.
        /// You must write an algorithm with O(log n) runtime complexity.
        /// </summary>
        /// <param name="nums"></param>
        /// <param name="target"></param>
        /// <returns></returns>
        public static int SearchInsert(int[] nums, int target)
        {
            if (nums.Length == 0) return 0;
            if (target == 0) return 0;

            int left = 0, right = nums.Length - 1;

            while (left <= right)
            {
                int mid = left + (right - left) / 2;

                if (nums[mid] == target)
                {
                    return mid; // Target found
                }
                else if (nums[mid] < target)
                {
                    left = mid + 1; // Search in the right half
                }
                else
                {
                    right = mid - 1; // Search in the left half
                }
            }

            return left;

        }

        /// <summary>
        /// Write an algorithm to determine if a number n is happy.
        /// A happy number is a number defined by the following process:
        /// Starting with any positive integer, replace the number by the sum of the squares of its digits.
        /// Repeat the process until the number equals 1 (where it will stay), or it loops endlessly in a cycle which does not include 1.
        /// Those numbers for which this process ends in 1 are happy.
        /// </summary>
        /// <param name="n"></param>
        /// <returns></returns>
        public static bool IsHappy(int n)
        {
            if (n == 0) return false;

            if (n < 10)
            {
                return n == 1 || n == 7;
            }
            int sum = 0;
            while (n > 0)
            {
                int digit = n % 10;
                sum += digit * digit;
                n /= 10;
            }

            return true;
        }

        /// <summary>
        /// Given two strings s and t, return true if t is an anagram of s, and
        /// false otherwise. Anagram being all the same characters are present
        /// even if the words are rearranged. 
        /// </summary>
        /// <param name="s"></param>
        /// <param name="t"></param>
        /// <returns></returns>
        public static bool IsAnagram(string s, string t)
        {
            if (string.IsNullOrEmpty(s)) return false;
            if (string.IsNullOrEmpty(t)) return false;
            if (s.Length != t.Length) return false;

            int[] saphabet = new int[26];
            foreach (char c in s)
            {
                saphabet[c - 'a']++;
            }
            foreach (char c in t)
            {
                if (saphabet[c - 'a'] > 0)
                {
                    saphabet[c - 'a']--;
                }
                else
                {
                    return false;
                }
            }
            return true;
        }

        /// <summary>
        /// Given a pattern and a string s, find if s follows the same pattern.
        /// Here follow means a full match, such that there is a bijection (1:1 mapping)
        /// between a letter in pattern and a non-empty word in s.
        /// Specifically:
        /// Each letter in pattern maps to exactly one unique word in s.
        /// Each unique word in s maps to exactly one letter in pattern.
        /// No two letters map to the same word, and no two words map to the same letter.
        /// </summary>
        /// <param name="pattern"></param>
        /// <param name="s"></param>
        /// <returns></returns>
        public static bool WordPattern(string pattern, string s)
        {
            if (string.IsNullOrEmpty(pattern)) return false;
            if (string.IsNullOrEmpty(s)) return false;

            var dictionary = new Dictionary<char, string>();
            var words = s.Split(' ');
            if (words.Length != pattern.Length)
            {
                return false;
            }
            for (var i = 0; i < pattern.Length; i++)
            {
                if (dictionary.TryGetValue(pattern[i], out var currentPair))
                {
                    if (currentPair != words[i])
                    {
                        return false;
                    }
                    continue;
                }
                if (dictionary.ContainsValue(words[i]))
                {
                    return false;
                }
                dictionary.Add(pattern[i], words[i]);
            }

            return true;
        }

        /// <summary>
        /// Give two strings s and t, determine if they are isomorphic.
        /// Two strings s and t are isomorphic if the characters in s can be replaced to get t.
        /// All occurrences of a character must be replaced with another character while preserving
        /// the order of characters. No two characters may map to the same character, but a character
        /// may map itself. 
        /// </summary>
        /// <param name="s"></param>
        /// <param name="t"></param>
        /// <returns></returns>
        public static bool IsIsomorphic(string s, string t)
        {
            if (string.IsNullOrEmpty(s)) return false;
            if (string.IsNullOrEmpty(t)) return false;

            if (s.Length != t.Length) return false;

            Dictionary<char, char> dict = new Dictionary<char, char>();
            for (int i = 0; i < s.Length; i++)
            {
                if (dict.TryGetValue(s[i], out char mappedChar))
                {
                    if (mappedChar != t[i])
                        return false;
                }
                else
                {
                    if (dict.ContainsValue(t[i]))
                        return false;
                    dict[s[i]] = t[i];
                }
            }

            return true;
        }

        /// <summary>
        /// Given two strings ransomNote and magazine, return true if ransomNote can be
        /// constructed by using the letters from magazine and false otherwise.
        /// </summary>
        /// <param name="ransomNote"></param>
        /// <param name="magazine"></param>
        /// <returns></returns>
        public static bool CanConstruct(string ransomNote, string magazine)
        {
            if (string.IsNullOrEmpty(ransomNote)) return false;
            if (string.IsNullOrEmpty(magazine)) return false;

            int[] letters = new int[26];
            foreach (char c in magazine)
            {
                letters[c - 'a']++;
            }

            foreach (char ch in ransomNote)
            {
                letters[ch - 'a']--;
                if (letters[ch - 'a'] == -1)
                    return false;
            }

            return true;
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