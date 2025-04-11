using System.Collections.Generic;
using System;
using System.ComponentModel;
using System.Data.Common;
using System.Data.SqlTypes;
using System.Diagnostics.Contracts;
using System.Diagnostics.Metrics;
using System.Drawing;
using System.IO;
using System.Linq.Expressions;
using System.Security.Cryptography;
using System.Text;
using System.Xml.Linq;
using static System.Net.Mime.MediaTypeNames;
using static System.Runtime.InteropServices.JavaScript.JSType;
using Microsoft.VisualBasic;
using System.Reflection;
using System.Numerics;
using System.Diagnostics;
using System.Text.RegularExpressions;
using System.Runtime.InteropServices;
using System.Runtime.ConstrainedExecution;
using static ConsoleApp1.Solution;
using System.Reflection.Emit;
using System.Reflection.Metadata;
using System.Xml;
using System.Net.NetworkInformation;
using System.Globalization;

namespace ConsoleApp1
{
    public class Node3
    {
        public int val;
        public IList<Node3> neighbors;

        public Node3()
        {
            val = 0;
            neighbors = new List<Node3>();
        }

        public Node3(int _val)
        {
            val = _val;
            neighbors = new List<Node3>();
        }

        public Node3(int _val, List<Node3> _neighbors)
        {
            val = _val;
            neighbors = _neighbors;
        }
    }

    public class Node
    {
        public int val;
        public Node next;
        public Node random;
        public Node(int _val)
        {
            val = _val;
            next = null;
            random = null;
        }
    }

    public class NodeC
    {
        public int val;
        public NodeC left;
        public NodeC right;
        public NodeC next;

        public NodeC() { }

        public NodeC(int _val)
        {
            val = _val;
        }

        public NodeC(int _val, NodeC _left, NodeC _right, NodeC _next)
        {
            val = _val;
            left = _left;
            right = _right;
            next = _next;
        }
    }

    public class NodeD
    {
        public bool val;
        public bool isLeaf;
        public NodeD topLeft;
        public NodeD topRight;
        public NodeD bottomLeft;
        public NodeD bottomRight;

        public NodeD()
        {
            val = false;
            isLeaf = false;
            topLeft = null;
            topRight = null;
            bottomLeft = null;
            bottomRight = null;
        }

        public NodeD(bool _val, bool _isLeaf)
        {
            val = _val;
            isLeaf = _isLeaf;
            topLeft = null;
            topRight = null;
            bottomLeft = null;
            bottomRight = null;
        }

        public NodeD(bool _val, bool _isLeaf, NodeD _topLeft, NodeD _topRight, NodeD _bottomLeft, NodeD _bottomRight)
        {
            val = _val;
            isLeaf = _isLeaf;
            topLeft = _topLeft;
            topRight = _topRight;
            bottomLeft = _bottomLeft;
            bottomRight = _bottomRight;
        }
    }

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

    //Definition for a binary tree node.
    public class TreeNode
    {
        public int val;
        public TreeNode left;
        public TreeNode right;
        public TreeNode(int val=0, TreeNode left = null, TreeNode right = null) {
            this.val = val;
            this.left = left;
            this.right = right;
        }
    }

    /// <summary>
    /// Main Program 2024-2025 leetcode grind... 
    /// </summary>
    public class Solution
    {
        private static Dictionary<string, Func<int, int, int>> s_Funcs = new() {
              { "+", (a, b) => a + b },
              { "-", (a, b) => b - a },
              { "*", (a, b) => a * b },
              { "/", (a, b) => b / a },
        };

        private static TreeNode result;

        private int i;
        private int nr;
        private int nc;
        private Dictionary<int, int> inorderMap;
        private List<int> RightSide { get; set; }
        private int Depth { get; set; }

        public static void Main()
        {
            Console.WriteLine(LongestConsecutive([0, 3, 7, 2, 5, 8, 4, 6, 0, 1]));
        }

        /// <summary>
        /// Given a string s, return the longest palindromic substring in s.
        /// </summary>
        /// <param name="s"></param>
        /// <returns></returns>
        public string LongestPalindrome(string s)
        {
            string T = "^#" + string.Join("#", s.ToCharArray()) + "#$";
            int n = T.Length;
            int[] P = new int[n];
            int C = 0, R = 0;

            for (int i = 1; i < n - 1; i++)
            {
                P[i] = (R > i) ? Math.Min(R - i, P[2 * C - i]) : 0;
                while (T[i + 1 + P[i]] == T[i - 1 - P[i]])
                    P[i]++;

                if (i + P[i] > R)
                {
                    C = i;
                    R = i + P[i];
                }
            }

            int max_len = P.Max();
            int center_index = Array.IndexOf(P, max_len);
            return s.Substring((center_index - max_len) / 2, max_len);
        }

        /// <summary>
        /// You are given an m x n integer array grid. There is a robot initially located
        /// at the top-left corner (i.e., grid[0][0]). The robot tries to move to the bottom-right
        /// corner (i.e., grid[m - 1][n - 1]). The robot can only move either down or right at any
        /// point in time. An obstacle and space are marked as 1 or 0 respectively in grid.
        /// A path that the robot takes cannot include any square that is an obstacle. Return the 
        /// number of possible unique paths that the robot can take to reach the bottom-right corner.
        /// The testcases are generated so that the answer will be less than or equal to 2 * 109.
        /// </summary>
        /// <param name="obstacleGrid"></param>
        /// <returns></returns>
        public int UniquePathsWithObstacles(int[][] obstacleGrid)
        {
            if (obstacleGrid == null || obstacleGrid.Length == 0 || obstacleGrid[0].Length == 0 || obstacleGrid[0][0] == 1)
            {
                return 0;
            }

            int m = obstacleGrid.Length;
            int n = obstacleGrid[0].Length;

            int[] previous = new int[n];
            int[] current = new int[n];
            previous[0] = 1;

            for (int i = 0; i < m; i++)
            {
                current[0] = obstacleGrid[i][0] == 1 ? 0 : previous[0];
                for (int j = 1; j < n; j++)
                {
                    current[j] = obstacleGrid[i][j] == 1 ? 0 : current[j - 1] + previous[j];
                }
                Array.Copy(current, previous, n);
            }

            return previous[n - 1];
        }

        /// <summary>
        /// Given a m x n grid filled with non-negative numbers, find a path from top
        /// left to bottom right, which minimizes the sum of all numbers along its path.
        /// Note: You can only move either down or right at any point in time.
        /// </summary>
        /// <param name="grid"></param>
        /// <returns></returns>
        public int MinPathSum(int[][] grid)
        {
            for (int i = 0; i < grid.Length; i++)
            {
                for (int j = 0; j < grid[i].Length; j++)
                {
                    if (i == 0 && j == 0)
                        continue;

                    if (i == 0)
                    {
                        grid[i][j] += grid[i][j - 1];
                        continue;
                    }

                    if (j == 0)
                    {
                        grid[i][j] += grid[i - 1][j];
                        continue;
                    }

                    grid[i][j] += Math.Min(grid[i][j - 1], grid[i - 1][j]);
                }
            }

            return grid[grid.Length - 1][grid[0].Length - 1];
        }

        /// <summary>
        /// Given a triangle array, return the minimum path sum from top to bottom.
        /// For each step, you may move to an adjacent number of the row below.More formally,
        /// if you are on index i on the current row, you may move to either index i or index i + 1 on the next row.
        /// </summary>
        /// <param name="triangle"></param>
        /// <returns></returns>
        public int MinimumTotal(IList<IList<int>> triangle)
        {
            if (triangle == null || triangle.Count == 0)
                return 0;

            int n = triangle.Count;
            int[] dp = new int[n + 1]; // Create a dp array to store intermediate results

            for (int row = n - 1; row >= 0; row--)
            {
                IList<int> currentRow = triangle[row];
                for (int i = 0; i < currentRow.Count; i++)
                {
                    dp[i] = Math.Min(dp[i], dp[i + 1]) + currentRow[i];
                }
            }

            return dp[0];
        }

        /// <summary>
        /// Given an integer array nums, return the length of the longest strictly increasing subsequence.
        /// </summary>
        /// <param name="nums"></param>
        /// <returns></returns>
        public int LengthOfLIS(int[] nums)
        {
            var len = 0;
            var tails = new int[nums.Length];

            foreach (int num in nums)
            {
                int left = 0, right = len;

                while (left < right)
                {
                    var mid = left + (right - left) / 2;
                    if (tails[mid] < num)
                    {
                        left = mid + 1;
                    }
                    else
                    {
                        right = mid;
                    }
                }

                tails[left] = num;
                if (left == len)
                {
                    len++;
                }
            }

            return len;
        }

        /// <summary>
        /// You are given an integer array coins representing coins of different denominations
        /// and an integer amount representing a total amount of money. Return the fewest number 
        /// of coins that you need to make up that amount.If that amount of money cannot be 
        /// made up by any combination of the coins, return -1. You may assume that you have 
        /// an infinite number of each kind of coin.
        /// </summary>
        /// <param name="coins"></param>
        /// <param name="amount"></param>
        /// <returns></returns>
        public int CoinChange(int[] coins, int amount)
        {
            //Build DP array
            int[] dp = new int[amount + 1];
            //Fill array with Amount + 1
            for (int i = 0; i < dp.Length; i++)
            {
                dp[i] = amount + 1;
            }
            //Base case
            dp[0] = 0;

            //Outer loop is for coins
            foreach (int coin in coins)
            {
                //Iterate over DP array to find how many coins needed to form that total of amount
                //Start from the denomination of coin as it's not possible to make total less than the denomination with this coin
                for (int i = coin; i <= amount; i++)
                {
                    //Update the DP array with minimum number of coins needed to make the total
                    dp[i] = Math.Min(dp[i], dp[i - coin] + 1);
                }
            }
            //Catch condition where if its not possible to form total with given coins return -1
            return dp[amount] <= amount ? dp[amount] : -1;
        }

        /// <summary>
        /// Given a string s and a dictionary of strings wordDict, return true if s can be segmented 
        /// into a space-separated sequence of one or more dictionary words.
        /// Note that the same word in the dictionary may be reused multiple times in the segmentation.
        /// </summary>
        /// <param name="s"></param>
        /// <param name="wordDict"></param>
        /// <returns></returns>
        public bool WordBreak(string s, IList<string> wordDict)
        {
            int n = s.Length;
            bool[] dp = new bool[n + 1];
            dp[0] = true;
            int max_len = 0;
            foreach (string word in wordDict)
            {
                max_len = Math.Max(max_len, word.Length);
            }

            for (int i = 1; i <= n; i++)
            {
                for (int j = i - 1; j >= Math.Max(i - max_len - 1, 0); j--)
                {
                    if (dp[j] && wordDict.Contains(s.Substring(j, i - j)))
                    {
                        dp[i] = true;
                        break;
                    }
                }
            }

            return dp[n];
        }

        /// <summary>
        /// You are a professional robber planning to rob houses along a street.
        /// Each house has a certain amount of money stashed, the only constraint 
        /// stopping you from robbing each of them is that adjacent houses have security
        /// systems connected and it will automatically contact the police if two adjacent 
        /// houses were broken into on the same night. Given an integer array nums representing
        /// the amount of money of each house, return the maximum amount of money you can rob 
        /// tonight without alerting the police.
        /// </summary>
        /// <param name="nums"></param>
        /// <returns></returns>
        public int Rob(int[] nums)
        {
            return TryRob(0, nums);
        }
        Dictionary<int, int> cache = new();
        private int TryRob(int idx, int[] nums)
        {
            if (idx >= nums.Length) return 0;

            if (cache.ContainsKey(idx))
                return cache[idx];

            int a = TryRob(idx + 2, nums) + nums[idx];
            int b = TryRob(idx + 1, nums);

            cache.Add(idx, Math.Max(a, b));
            return Math.Max(a, b);
        }

        /// <summary>
        /// Implement pow(x, n), which calculates x raised to the power n (i.e., xn).
        /// </summary>
        /// <param name="x"></param>
        /// <param name="n"></param>
        /// <returns></returns>
        public double MyPow(double x, int n)
        {
            if (n < 0)
            {
                x = 1 / x;
                n = -n;
            }

            double result = 1;
            double current_product = x;

            while (n > 0)
            {
                if (n % 2 == 1)
                {
                    result = result * current_product;
                }
                current_product = current_product * current_product;
                n = n / 2;
            }

            return result;
        }

        /// <summary>
        /// Given an integer n, return the number of trailing zeroes in n!.
        /// Note that n! = n* (n - 1) * (n - 2) * ... * 3 * 2 * 1.
        /// </summary>
        /// <param name="n"></param>
        /// <returns></returns>
        public int TrailingZeroes(int n)
        {
            // we have to find counts of 5, 25, 125, 625, etc. dividends in n
            int trailingZeroes = 0;

            while (n >= 5)
            {
                n /= 5;
                trailingZeroes += n;
            }

            return trailingZeroes;
        }

        /// <summary>
        /// You are given an array of variable pairs equations and an array of real numbers values, 
        /// where equations[i] = [Ai, Bi] and values[i] represent the equation Ai / Bi = values[i]. 
        /// Each Ai or Bi is a string that represents a single variable. You are also given some queries,
        /// where queries[j] = [Cj, Dj] represents the jth query where you must find the answer for Cj / Dj = ?.
        /// Return the answers to all queries.If a single answer cannot be determined, return -1.0.
        /// Note: The input is always valid. You may assume that evaluating the queries will not result
        /// in division by zero and that there is no contradiction. Note: The variables that do not occur in
        /// the list of equations are undefined, so the answer cannot be determined for them.
        /// </summary>
        /// <param name="equations"></param>
        /// <param name="values"></param>
        /// <param name="queries"></param>
        /// <returns></returns>
        public double[] CalcEquation(IList<IList<string>> eq, double[] vals, IList<IList<string>> q)
        {
            Dictionary<string, Dictionary<string, double>> map = new();
            HashSet<string> visited = new();

            foreach (var (num, den, val) in eq.Zip(vals, (e, v) => (e[0], e[1], v)))
            {
                if (!map.ContainsKey(num)) map[num] = new();
                if (!map.ContainsKey(den)) map[den] = new();

                map[num][den] = 1 / val;
                map[den][num] = val;
            }

            return q.Select(s => FindResult(s[1], s[0])).ToArray();

            double FindResult(string s, string t)
            {
                if (!map.ContainsKey(s)) return -1;
                if (s == t) return 1;

                double cur = -1;
                visited.Add(s);

                foreach (var k in map[s].Keys)
                {
                    if (visited.Contains(k)) continue;
                    cur = FindResult(k, t);
                    if (cur != -1)
                    {
                        cur *= map[s][k];
                        break;
                    }
                }

                visited.Remove(s);
                return cur;
            }
        }

        /// <summary>
        /// Given a reference of a node in a connected undirected graph.
        /// Return a deep copy(clone) of the graph.
        /// Each node in the graph contains a value(int) and a list(List[Node]) of its neighbors.
        /// </summary>
        /// <param name="node"></param>
        /// <returns></returns>
        Dictionary<Node3, Node3> reference = new Dictionary<Node3, Node3>();

        public bool CheckClone(Node3 node)
        {
            if (reference.ContainsKey(node)) return true;

            reference.Add(node, new Node3(node.val));
            return false;
        }

        public void DeepClone(Node3 node)
        {
            if (CheckClone(node)) return;

            for (int i = 0; i < node.neighbors.Count; i++)
            {
                DeepClone(node.neighbors[i]);
                reference[node].neighbors.Add(reference[node.neighbors[i]]);
            }
        }

        public Node3 CloneGraph(Node3 node)
        {
            if (node == null) return null;

            DeepClone(node);

            return reference[node];
        }

        /// <summary>
        /// You are given an m x n matrix board containing letters 'X' and 'O', capture regions that are surrounded:
        /// Connect: A cell is connected to adjacent cells horizontally or vertically. Region: To form a region connect 
        /// every 'O' cell. Surround: The region is surrounded with 'X' cells if you can connect the region with 'X' 
        /// cells and none of the region cells are on the edge of the board. To capture a surrounded region, replace 
        /// all 'O's with 'X's in-place within the original board.You do not need to return anything.
        /// </summary>
        /// <param name="board"></param>
        public void Solve(char[][] board)
        {
            int m = board.Length;
            int n = board[0].Length;

            // Mark all border-connected 'O's as 'T'
            for (int i = 0; i < m; i++)
            {
                if (board[i][0] == 'O') DFS(board, i, 0);
                if (board[i][n - 1] == 'O') DFS(board, i, n - 1);
            }
            for (int j = 0; j < n; j++)
            {
                if (board[0][j] == 'O') DFS(board, 0, j);
                if (board[m - 1][j] == 'O') DFS(board, m - 1, j);
            }

            // Flip all 'O's to 'X' and 'T's back to 'O'
            for (int i = 0; i < m; i++)
            {
                for (int j = 0; j < n; j++)
                {
                    board[i][j] = board[i][j] != 'T' ? 'X' : 'O';
                }
            }
        }


        /// <summary>
        /// Given an m x n 2D binary grid grid which represents a map of '1's (land) and '0's (water), 
        /// return the number of islands. An island is surrounded by water and is formed by connecting 
        /// adjacent lands horizontally or vertically.You may assume all four edges of the grid are all 
        /// surrounded by water.
        /// </summary>
        /// <param name="grid"></param>
        /// <returns></returns>
        public int NumIslands(char[][] grid)
        {
            var islands = 0;

            // Traverse the grid one by one
            for (int i = 0; i < grid.Length; i++)
            {
                for (int j = 0; j < grid[0].Length; j++)
                {
                    // If grid value is '1', increment the count, and do DFS on the adjacent 4 cells
                    if (grid[i][j] == '1')
                    {
                        DFS(grid, i, j);
                        islands++;
                    }
                }
            }
            // Return the island count
            return islands;
        }

        /// <summary>
        /// Given the root of a binary tree, determine if it is a valid binary search tree (BST).
        /// A valid BST is defined as follows: The left subtree of a node contains only nodes 
        /// with keys less than the node's key. The right subtree of a node contains only nodes 
        /// with keys greater than the node's key. Both the left and right subtrees must also be
        /// binary search trees.
        /// </summary>
        /// <param name="root"></param>
        /// <returns></returns>
        public bool IsValidBST(TreeNode root)
        {
            return Evaluate(root, long.MinValue, long.MaxValue);
        }

        private bool Evaluate(TreeNode node, long min, long max)
        {
            if (node == null)
            {
                return true;
            }

            return (
                node.val > min &&
                node.val < max &&
                Evaluate(node.left, min, node.val) &&
                Evaluate(node.right, node.val, max)
            );
        }

        /// <summary>
        /// You are given two integer arrays nums1 and nums2 sorted in non-decreasing order and an integer k.
        /// Define a pair(u, v) which consists of one element from the first array and one element from the second array.
        /// Return the k pairs (u1, v1), (u2, v2), ..., (uk, vk) with the smallest sums.
        /// </summary>
        /// <param name="nums1"></param>
        /// <param name="nums2"></param>
        /// <param name="k"></param>
        /// <returns></returns>
        public IList<IList<int>> KSmallestPairs(int[] nums1, int[] nums2, int k)
        {
            var result = new List<IList<int>>();
            if (nums1 == null || nums1.Length == 0 || nums2 == null || nums2.Length == 0)
                return result;

            // Create a min heap to store pairs based on their sums
            SortedSet<(int sum, int index1, int index2)> minHeap = new SortedSet<(int, int, int)>(Comparer<(int, int, int)>.Create((a, b) =>
            {
                int compare = a.Item1.CompareTo(b.Item1);
                if (compare == 0)
                    compare = a.Item2.CompareTo(b.Item2);
                if (compare == 0)
                    compare = a.Item3.CompareTo(b.Item3);
                return compare;
            }));

            // Add the first pair (nums1[0], nums2[0]) to the min heap
            minHeap.Add((nums1[0] + nums2[0], 0, 0));

            while (k > 0 && minHeap.Count > 0)
            {
                // Get the pair with the smallest sum from the min heap
                var (sum, index1, index2) = minHeap.Min;
                minHeap.Remove(minHeap.Min);

                // Add the pair to the result
                result.Add(new List<int> { nums1[index1], nums2[index2] });

                // Explore the next pair options by moving the indices
                if (index1 < nums1.Length - 1)
                    minHeap.Add((nums1[index1 + 1] + nums2[index2], index1 + 1, index2));

                if (index1 == 0 && index2 < nums2.Length - 1)
                    minHeap.Add((nums1[index1] + nums2[index2 + 1], index1, index2 + 1));

                k--;
            }

            return result;
        }

        /// <summary>
        /// Given an integer array nums and an integer k, return the kth largest element in the array.
        /// Note that it is the kth largest element in the sorted order, not the kth distinct element.
        /// Can you solve it without sorting?
        /// </summary>
        /// <param name="nums"></param>
        /// <param name="k"></param>
        /// <returns></returns>
        public int FindKthLargest(int[] nums, int k)
        {
            Suffle(nums);
            Quick3WaySort(nums, 0, nums.Length - 1, k);
            return nums[k - 1];
        }

        private void Suffle(int[] nums)
        {
            var random = new Random();
            int N = nums.Length;
            int r, temp;
            for (int i = 0; i < N; i++)
            {
                r = random.Next(i + 1);

                temp = nums[r];
                nums[r] = nums[i];
                nums[i] = temp;
            }
        }

        private void Swap(int[] nums, int i, int j)
        {
            var temp = nums[i];
            nums[i] = nums[j];
            nums[j] = temp;
        }

        private void Quick3WaySort(int[] nums, int lo, int hi, int k)
        {
            if (lo >= hi) return;
            if (lo >= k) return;
            if (hi < k - 1) return;

            int lt = hi, gt = lo, i = lo;
            int pivot = nums[i];
            while (i <= lt)
            {
                if (nums[i] > pivot)
                    Swap(nums, gt++, i);
                else if (nums[i] < pivot)
                    Swap(nums, lt--, i);
                else
                    i++;
            }

            Quick3WaySort(nums, lo, gt - 1, k);
            Quick3WaySort(nums, lt + 1, hi, k);
        }

        /// <summary>
        /// Suppose an array of length n sorted in ascending order is rotated between 1 and n times. 
        /// For example, the array nums = [0,1,2,4,5,6,7] might become: [4, 5, 6, 7, 0, 1, 2] if it was rotated 4 times.
        /// [0, 1, 2, 4, 5, 6, 7] if it was rotated 7 times. Notice that rotating an array[a[0], a[1], a[2], ..., a[n - 1]] 
        /// 1 time results in the array[a[n - 1], a[0], a[1], a[2], ..., a[n - 2]]. Given the sorted rotated array nums of
        /// unique elements, return the minimum element of this array. You must write an algorithm that runs in O(log n) time.
        /// </summary>
        /// <param name="nums"></param>
        /// <returns></returns>
        public int FindMin(int[] nums)
        {
            // Initialize left and right pointers
            int left = 0, right = nums.Length - 1;

            // Binary Search
            while (left < right)
            {
                // Calculate mid-point
                int mid = left + (right - left) / 2;

                // If mid-point element is greater than the last element
                // of the array, then the minimum element must be in the
                // right half of the array, so we update left pointer
                if (nums[mid] > nums[right])
                {
                    left = mid + 1;
                }
                // Otherwise, the minimum element must be in the left half
                // of the array, so we update right pointer
                else
                {
                    right = mid;
                }
            }

            // At the end of the while loop, left pointer points to the
            // minimum element of the array, which is the answer
            return nums[left];
        }

        /// <summary>
        /// Given an array of integers nums sorted in non-decreasing order, find the starting and ending 
        /// position of a given target value. If target is not found in the array, return [-1, -1].
        /// You must write an algorithm with O(log n) runtime complexity.
        /// </summary>
        /// <param name="nums"></param>
        /// <param name="target"></param>
        /// <returns></returns>
        public int[] SearchRange(int[] nums, int target)
        {
            int lo = 0, hi = nums.Length - 1, mid;
            while (lo <= hi)
            {
                mid = lo + (hi - lo) / 2;
                if (target > nums[mid])
                    lo = mid + 1;
                else
                    hi = mid - 1;
            }
            if (lo == nums.Length || nums[lo] != target)
                return new int[] { -1, -1 };

            var result = new int[2];
            result[0] = lo;

            lo = 0; hi = nums.Length - 1;
            while (lo <= hi)
            {
                mid = lo + (hi - lo) / 2;
                if (target >= nums[mid])
                    lo = mid + 1;
                else
                    hi = mid - 1;
            }
            result[1] = lo - 1;

            return result;
        }

        /// <summary>
        /// There is an integer array nums sorted in ascending order (with distinct values).
        /// Prior to being passed to your function, nums is possibly rotated at an unknown
        /// pivot index k(1 <= k<nums.length) such that the resulting array is [nums[k], nums[k + 1]
        /// , ..., nums[n - 1], nums[0], nums[1], ..., nums[k - 1]] (0-indexed). For example, [0, 1,
        /// 2, 4, 5, 6, 7] might be rotated at pivot index 3 and become[4, 5, 6, 7, 0, 1, 2].
        /// Given the array nums after the possible rotation and an integer target, return the 
        /// index of target if it is in nums, or -1 if it is not in nums.
        /// You must write an algorithm with O(log n) runtime complexity.
        /// </summary>
        /// <param name="nums"></param>
        /// <param name="target"></param>
        /// <returns></returns>
        public int Search(int[] nums, int target)
        {
            int low = 0, high = nums.Length - 1;

            while (low <= high)
            {
                int mid = (low + high) / 2;

                if (nums[mid] == target)
                {
                    return mid;
                }

                if (nums[low] <= nums[mid])
                {
                    if (nums[low] <= target && target < nums[mid])
                    {
                        high = mid - 1;
                    }
                    else
                    {
                        low = mid + 1;
                    }
                }
                else
                {
                    if (nums[mid] < target && target <= nums[high])
                    {
                        low = mid + 1;
                    }
                    else
                    {
                        high = mid - 1;
                    }
                }
            }

            return -1;
        }

        /// <summary>
        /// A peak element is an element that is strictly greater than its neighbors.
        /// Given a 0-indexed integer array nums, find a peak element, and return its index.
        /// If the array contains multiple peaks, return the index to any of the peaks.
        /// You may imagine that nums[-1] = nums[n] = -∞. In other words, an element is always 
        /// considered to be strictly greater than a neighbor that is outside the array.
        /// You must write an algorithm that runs in O(log n) time.
        /// </summary>
        /// <param name="nums"></param>
        /// <returns></returns>
        public int FindPeakElement(int[] nums)
        {
            var left = 0;
            var right = nums.Length - 1;

            while (left + 1 < right)
            {
                var mid = left + (right - left) / 2;

                if (nums[mid] < nums[mid + 1])
                {
                    left = mid;
                }
                else
                {
                    right = mid;
                }
            }

            return nums[left] > nums[right] ? left : right;
        }

        /// <summary>
        /// You are given an m x n integer matrix matrix with the following two properties:
        /// Each row is sorted in non-decreasing order. The first integer of each row is greater
        /// than the last integer of the previous row. Given an integer target, return true if 
        /// target is in matrix or false otherwise. You must write a solution in O(log(m* n)) time complexity.
        /// </summary>
        /// <param name="matrix"></param>
        /// <param name="target"></param>
        /// <returns></returns>
        public bool SearchMatrix(int[][] matrix, int target)
        {
            int m = matrix.Length;
            int n = matrix[0].Length;
            int left = 0, right = m * n - 1;

            while (left <= right)
            {
                int mid = left + (right - left) / 2;
                int mid_val = matrix[mid / n][mid % n];

                if (mid_val == target)
                    return true;
                else if (mid_val < target)
                    left = mid + 1;
                else
                    right = mid - 1;
            }
            return false;
        }

        /// <summary>
        /// Given a circular integer array nums of length n, return the maximum possible sum of a non-empty subarray of nums.
        /// A circular array means the end of the array connects to the beginning of the array.Formally, the next element of 
        /// nums[i] is nums[(i + 1) % n] and the previous element of nums[i] is nums[(i - 1 + n) % n]. A subarray may only 
        /// include each element of the fixed buffer nums at most once.Formally, for a subarray nums[i], nums[i + 1], ..., 
        /// nums[j], there does not exist i <= k1, k2 <= j with k1 % n == k2 % n.
        /// </summary>
        /// <param name="nums"></param>
        /// <returns></returns>
        public int MaxSubarraySumCircular(int[] nums)
        {
            int total = 0, maxSum = -30000, curMax = 0, minSum = 30000, curMin = 0;
            foreach (int a in nums)
            {
                curMax = Math.Max(curMax + a, a);
                maxSum = Math.Max(maxSum, curMax);
                curMin = Math.Min(curMin + a, a);
                minSum = Math.Min(minSum, curMin);
                total += a;
            }
            return maxSum > 0 ? Math.Max(maxSum, total - minSum) : maxSum;
        }

        /// <summary>
        /// Given an integer array nums, find the subarray with the largest sum, and return its sum.
        /// </summary>
        /// <param name="nums"></param>
        /// <returns></returns>
        public int MaxSubArray(int[] nums)
        {
            int maxSum = nums[0];
            int curSum = 0;

            foreach (int num in nums)
            {
                curSum = Math.Max(0, curSum);
                curSum += num;
                maxSum = Math.Max(curSum, maxSum);
            }
            return maxSum;
        }

        /// <summary>
        /// Given a n * n matrix grid of 0's and 1's only. We want to represent grid with a Quad-Tree.
        /// Return the root of the Quad-Tree representing grid.
        /// </summary>
        /// <param name="grid"></param>
        /// <returns></returns>
        public NodeD Construct(int[][] grid)
            => GetNode(grid, grid.Length, 0, 0);

        private NodeD GetNode(int[][] grid, int size, int i, int k)
        {
            if (size == 1 || AreSame(grid, size, i, k))
                return new NodeD(grid[i][k] == 1, true);

            return new NodeD(true, false,
                GetNode(grid, size / 2, i, k),
                GetNode(grid, size / 2, i, k + (size / 2)),
                GetNode(grid, size / 2, i + (size / 2), k),
                GetNode(grid, size / 2, i + (size / 2), k + (size / 2)));
        }

        private bool AreSame(int[][] grid, int size, int i, int k)
        {
            for (int ii = i; ii < i + size; ii++)
                for (int kk = k; kk < k + size; kk++)
                    if (grid[ii][kk] != grid[i][k])
                        return false;

            return true;
        }

        /// <summary>
        /// Given the head of a linked list, return the list after sorting it in ascending order.
        /// </summary>
        /// <param name="head"></param>
        /// <returns></returns>
        public ListNode SortList(ListNode head)
        {
            if (head is null || head.next is null)
            {
                return head;
            }

            var left = head;
            var right = GetMiddle(head);

            // we divide head linked list into 2 sublists
            var temp = right.next;
            right.next = null;
            right = temp;

            left = SortList(left);
            right = SortList(right);

            return MergeLists(left, right);
        }

        private ListNode GetMiddle(ListNode head)
        {
            var slow = head;
            var fast = head.next;

            while (fast is not null && fast.next is not null)
            {
                slow = slow.next;
                fast = fast.next.next;
            }

            return slow;
        }

        private ListNode MergeLists(ListNode first, ListNode second)
        {
            var dummy = new ListNode();
            var tail = dummy;

            while (first is not null && second is not null)
            {
                if (first.val < second.val)
                {
                    tail.next = first;
                    first = first.next;
                }
                else
                {
                    tail.next = second;
                    second = second.next;
                }

                tail = tail.next;
            }

            if (first is not null)
            {
                tail.next = first;
            }

            if (second is not null)
            {
                tail.next = second;
            }

            return dummy.next;
        }


        /// <summary>
        /// Given an m x n grid of characters board and a string word, return true if word exists in the grid.
        /// The word can be constructed from letters of sequentially adjacent cells, where adjacent cells are 
        /// horizontally or vertically neighboring.The same letter cell may not be used more than once.
        /// </summary>
        /// <param name="board"></param>
        /// <param name="word"></param>
        /// <returns></returns>
        public bool IsSafe(bool[,] vis, int i, int j, int m, int n)
        {
            return i >= 0 && j >= 0 && i < m && j < n && !vis[i, j];
        }

        public bool Rec(int i, int j, int index, int m, int n, char[][] board, string word, bool[,] vis)
        {
            if (index == word.Length) return true;
            if (!IsSafe(vis, i, j, m, n) || board[i][j] != word[index]) return false;

            vis[i, j] = true;

            bool left = Rec(i - 1, j, index + 1, m, n, board, word, vis);
            bool right = Rec(i + 1, j, index + 1, m, n, board, word, vis);
            bool up = Rec(i, j - 1, index + 1, m, n, board, word, vis);
            bool down = Rec(i, j + 1, index + 1, m, n, board, word, vis);

            vis[i, j] = false;

            return left || right || up || down;
        }

        public bool Exist(char[][] board, string word)
        {
            int m = board.Length, n = board[0].Length;
            for (int i = 0; i < m; i++)
            {
                for (int j = 0; j < n; j++)
                {
                    if (board[i][j] == word[0])
                    {
                        bool[,] vis = new bool[m, n];
                        if (Rec(i, j, 0, m, n, board, word, vis))
                        {
                            return true;
                        }
                    }
                }
            }
            return false;
        }

        /// <summary>
        /// Given n pairs of parentheses, write a function to generate all combinations of well-formed parentheses.
        /// </summary>
        /// <param name="n"></param>
        /// <returns></returns>
        public IList<string> GenerateParenthesis(int n)
        {
            var result = new List<string>();
            GenerateCombinations(result, "", 0, 0, n);
            return result;
        }

        private void GenerateCombinations(IList<string> result, string current, int open, int close, int max)
        {
            if (current.Length == max * 2)
            {
                result.Add(current);
                return;
            }

            if (open < max)
            {
                GenerateCombinations(result, current + "(", open + 1, close, max);
            }

            if (close < open)
            {
                GenerateCombinations(result, current + ")", open, close + 1, max);
            }
        }

        /// <summary>
        /// Given an array of distinct integers candidates and a target integer target, return a list 
        /// of all unique combinations of candidates where the chosen numbers sum to target. 
        /// You may return the combinations in any order. The same number may be chosen from candidates
        /// an unlimited number of times.Two combinations are unique if the frequency of at least one 
        /// of the chosen numbers is different. The test cases are generated such that the number of 
        /// unique combinations that sum up to target is less than 150 combinations for the given input.
        /// </summary>
        /// <param name="candidates"></param>
        /// <param name="target"></param>
        /// <returns></returns>
        public IList<IList<int>> CombinationSum(int[] candidates, int target)
        {
            IList<IList<int>> result = new List<IList<int>>();
            Array.Sort(candidates);
            FindUniqueCombinations(candidates, target, new List<int>(), result, 0);
            return result;
        }
        private void FindUniqueCombinations(int[] candidates, int rem, List<int> lst,
        IList<IList<int>> result, int idx)
        {
            if (rem == 0) result.Add(new List<int>(lst));
            else if (rem < 0) return;
            else
            {
                for (int i = idx; i < candidates.Length; i++)
                {
                    lst.Add(candidates[i]);
                    FindUniqueCombinations(candidates, rem - candidates[i], lst, result, i);
                    lst.RemoveAt(lst.Count - 1);
                }
            }
        }

        /// <summary>
        /// Given an array nums of distinct integers, return all the possible permutations.
        /// You can return the answer in any order.
        /// </summary>
        /// <param name="nums"></param>
        /// <returns></returns>
        public IList<IList<int>> Permute(int[] nums)
        {
            IList<IList<int>> result = new List<IList<int>>();
            Backtrack(nums, new List<int>(), result);
            return result;
        }

        private void Backtrack(int[] nums, List<int> path, IList<IList<int>> result)
        {
            if (path.Count == nums.Length)
            {
                result.Add(new List<int>(path));
                return;
            }
            foreach (int num in nums)
            {
                if (path.Contains(num)) continue;
                path.Add(num);
                Backtrack(nums, path, result);
                path.RemoveAt(path.Count - 1);
            }
        }

        /// <summary>
        /// Given two integers n and k, return all possible combinations of k numbers chosen from the range [1, n].
        /// You may return the answer in any order.
        /// </summary>
        /// <param name="n"></param>
        /// <param name="k"></param>
        /// <returns></returns>
        public IList<IList<int>> Combine(int n, int k)
        {
            var result = new List<IList<int>>();
            GenerateCombinations(1, n, k, new List<int>(), result);
            return result;
        }

        private void GenerateCombinations(int start, int n, int k, List<int> combination, IList<IList<int>> result)
        {
            if (k == 0)
            {
                result.Add(new List<int>(combination));
                return;
            }
            for (var i = start; i <= n; ++i)
            {
                combination.Add(i);
                GenerateCombinations(i + 1, n, k - 1, combination, result);
                combination.RemoveAt(combination.Count - 1);
            }
        }

        /// <summary>
        /// Given a string containing digits from 2-9 inclusive, return all possible letter combinations 
        /// that the number could represent. Return the answer in any order. A mapping of digits to 
        /// letters(just like on the telephone buttons) is given below.Note that 1 does not map to any letters.
        /// </summary>
        /// <param name="digits"></param>
        /// <returns></returns>
        public IList<string> LetterCombinations(string digits)
        {
            string[] phoneChars = new string[] { " ",
                                           "",
                                           "abc",
                                           "def",
                                           "ghi",
                                           "jkl",
                                           "mno",
                                           "pqrs",
                                           "tuv",
                                           "wxyz"
                                         };

            if (digits.Length == 0) return new List<string>();

            var results = new List<string>() { "" };
            foreach (var digit in digits)
            {
                var keys = phoneChars[digit - '0'];
                if (keys.Length == 0) continue;

                var temp = new List<string>();
                foreach (var result in results)
                    foreach (var ch in keys)
                        temp.Add(result + ch.ToString());

                results = temp;
            }

            if (results.Count == 1 && results[0] == "") results.Clear();
            return results;
        }

        /// <summary>
        /// Given an m x n board of characters and a list of strings words, return all words on the board.
        /// Each word must be constructed from letters of sequentially adjacent cells, where adjacent 
        /// cells are horizontally or vertically neighboring.The same letter cell may not be used more
        /// than once in a word.
        /// </summary>
        /// <param name="board"></param>
        /// <param name="words"></param>
        /// <returns></returns>
        public IList<string> FindWords(char[][] board, string[] words)
        {
            Dictionary<char, List<(int row, int column)>> lettersPositions = InitializeDictionary(board);
            List<string> answer = new(words.Length);

            bool shouldReverse = words.All(w => w[0] == words[0][0]);
            foreach (var word in words)
            {
                var reversed = shouldReverse ? Reverse(word) : word;
                //var possibleStartLocations = lettersPositions[word[0]];
                var possibleStartLocations = lettersPositions[reversed[0]];
                foreach (var possibility in possibleStartLocations)
                {
                    //if(WordStartsAt(word, board, possibility))
                    if (WordStartsAt(reversed, board, possibility))
                    {
                        answer.Add(word);
                        break;
                    }
                }
            }

            return answer;
        }

        public Dictionary<char, List<(int row, int column)>> InitializeDictionary(char[][] board)
        {
            Dictionary<char, List<(int row, int column)>> lettersPositions = new();
            for (int letter = 'a'; letter <= 'z'; letter++)
            {
                lettersPositions[(char)letter] = new();
            }

            // Add the array letter positions to the dictionary
            for (int row = 0; row < board.Length; row++)
            {
                for (int col = 0; col < board[row].Length; col++)
                {
                    lettersPositions[board[row][col]].Add((row, col));
                }
            }
            return lettersPositions;
        }

        public bool WordStartsAt(string word, char[][] board, (int row, int col) start)
        {
            Stack<(int row, int col)> positions = new();
            return CheckLetters(word, board, start, positions, 1);
        }

        public static string Reverse(string s)
        {
            char[] charArray = s.ToCharArray();
            Array.Reverse(charArray);
            return new string(charArray);
        }

        public bool CheckLetters(string word, char[][] board, (int row, int col) start, Stack<(int row, int col)> positions, int index)
        {
            if (word.Length == index)
                return true;

            positions.Push(start);
            char target = word[index];
            var neighbours = GetNeighbours(board, start);

            foreach (var neighbour in neighbours)
            {
                if (board[neighbour.row][neighbour.col] != target || positions.Contains(neighbour))
                    continue;

                if (CheckLetters(word, board, neighbour, positions, index + 1))
                    return true;
            }

            positions.Pop();
            return false;
        }

        public List<(int row, int col)> GetNeighbours(char[][] board, (int row, int col) start)
        {
            return new[]{
                (start.row + 1, start.col),
                (start.row - 1, start.col),
                (start.row, start.col + 1),
                (start.row, start.col - 1)}
                    .Where(x => IsInbound(board, x)).ToList();
        }

        public bool IsInbound(char[][] board, (int row, int col) pos)
        {
            return pos.row >= 0
                && pos.row < board.Length
                && pos.col >= 0
                && pos.col < board[0].Length;
        }

        /// <summary>
        /// Design a data structure that supports adding new words and finding if a 
        /// string matches any previously added string. Implement the WordDictionary class:
        /// WordDictionary() Initializes the object. void addWord(word) Adds word to the 
        /// data structure, it can be matched later. bool search(word) Returns true if there 
        /// is any string in the data structure that matches word or false otherwise.
        /// word may contain dots '.' where dots can be matched with any letter.
        /// </summary>
        public class WordDictionary
        {

            TrieNode root;
            public WordDictionary()
            {
                root = new TrieNode();
            }

            public void AddWord(string word)
            {
                var temp = root;
                foreach (char ch in word)
                {
                    if (temp.Nodes.ContainsKey(ch))
                    {
                        temp = temp.Nodes[ch];
                    }
                    else
                    {
                        var newNode = new TrieNode();
                        temp.Nodes.Add(ch, newNode);
                        temp = newNode;
                    }
                }

                temp.IsWordEnd = true;
            }

            public bool Search(string word)
            {
                return SearchTrie(root, word, 0);
            }

            private bool SearchTrie(TrieNode root, string word, int index)
            {
                TrieNode temp = root;

                if (index == word.Length)
                {
                    return temp.IsWordEnd;
                }

                if (word[index] != '.')
                {
                    if (!temp.Nodes.ContainsKey(word[index]))
                    {
                        return false;
                    }

                    temp = temp.Nodes[word[index]];
                    return SearchTrie(temp, word, index + 1);
                }
                else
                {
                    foreach (var node in temp.Nodes)
                    {
                        if (SearchTrie(node.Value, word, index + 1))
                        {
                            return true;
                        }
                    }
                }

                return false;
            }
        }

        public class TrieNode
        {
            public bool IsWordEnd;
            public Dictionary<char, TrieNode> Nodes;

            public TrieNode()
            {
                IsWordEnd = false;
                Nodes = new Dictionary<char, TrieNode>();
            }
        }

        /// <summary>
        /// A trie (pronounced as "try") or prefix tree is a tree data structure used to efficiently store 
        /// and retrieve keys in a dataset of strings. There are various applications of this data structure,
        /// such as autocomplete and spellchecker. Implement the Trie class: Trie() Initializes the trie object.
        /// void insert(String word) Inserts the string word into the trie. boolean search(String word) 
        /// Returns true if the string word is in the trie(i.e., was inserted before), and false otherwise.
        /// boolean startsWith(String prefix) Returns true if there is a previously inserted string word that
        /// has the prefix prefix, and false otherwise.
        /// </summary>
        public class Trie
        {
            HashSet<string> values;
            HashSet<string> keys;
            public Trie()
            {
                values = new HashSet<string>();
                keys = new HashSet<string>();
            }

            public void Insert(string word)
            {
                if (!values.Contains(word))
                    values.Add(word);
                for (var i = word.Length; i > 0; i--)
                {
                    if (!keys.Contains(word.Substring(0, i)))
                        keys.Add(word.Substring(0, i));
                    else
                        break;// all prev string are in KEYS
                }
            }

            public bool Search(string word)
            {
                return values.Contains(word);
            }

            public bool StartsWith(string prefix)
            {
                return keys.Contains(prefix);
            }
        }

        /// <summary>
        /// A gene string can be represented by an 8-character long string, with choices from 'A', 'C', 'G', and 'T'.
        /// Suppose we need to investigate a mutation from a gene string startGene to a gene string endGene where one 
        /// mutation is defined as one single character changed in the gene string. For example, "AACCGGTT" --> "AACCGGTA" 
        /// is one mutation. There is also a gene bank bank that records all the valid gene mutations. A gene must be in 
        /// bank to make it a valid gene string. Given the two gene strings startGene and endGene and the gene bank bank, 
        /// return the minimum number of mutations needed to mutate from startGene to endGene. If there is no such a mutation,
        /// return -1. Note that the starting point is assumed to be valid, so it might not be included in the bank.
        /// </summary>
        /// <param name="startGene"></param>
        /// <param name="endGene"></param>
        /// <param name="bank"></param>
        /// <returns></returns>
        public int MinMutation(string startGene, string endGene, string[] bank)
        {
            //if bank is empty, return
            if (bank.Length == 0)
            {
                return -1;
            }

            // keep track of what nodes we've seen
            HashSet<string> seen = new HashSet<string>();

            // queue of nodes to check
            Queue<GeneNode> wordQueue = new Queue<GeneNode>();

            // add starting gene
            seen.Add(startGene);

            // add first node to queue
            wordQueue.Enqueue(new GeneNode(startGene, 0));

            // while queue contains anything, keep searching
            while (wordQueue.Any())
            {

                // get next thing in the queue
                GeneNode currGene = wordQueue.Dequeue();

                // we found the end gene! yay! return how many steps it took
                if (currGene.currentGene == endGene)
                {
                    return currGene.mutations;
                }

                //go through each gene in the bank
                foreach (string bankGene in bank)
                {

                    // we already looked at this gene, move on
                    if (seen.Contains(bankGene)) { continue; }

                    // how many letters are different between these two genes
                    int letterCount = 0;

                    // go through each letter and check against the current genes letter
                    for (int x = 0; x < bankGene.Length; x++)
                    {
                        if (currGene.currentGene[x] != bankGene[x])
                        {
                            letterCount += 1;
                        }
                        //too many letters, end the loop
                        if (letterCount >= 2)
                        {
                            break;
                        }
                    }
                    if (letterCount <= 1)
                    {
                        wordQueue.Enqueue(new GeneNode(bankGene, currGene.mutations + 1));
                        seen.Add(bankGene);
                    }
                }
            }
            return -1;

        }
        // class to keep track of current BFS search
        private class GeneNode
        {
            // Current string
            public string currentGene;
            // Number of mutations it took to get here
            public int mutations;

            public GeneNode(string currentGene, int mutations)
            {
                this.currentGene = currentGene;
                this.mutations = mutations;
            }
        }

        /// <summary>
        /// You are given an n x n integer matrix board where the cells are labeled from 1 to n2 in a Boustrophedon style 
        /// starting from the bottom left of the board (i.e. board[n - 1][0]) and alternating direction each row.
        /// You start on square 1 of the board. In each move, starting from square curr, do the following:
        /// Choose a destination square next with a label in the range[curr + 1, min(curr + 6, n2)].
        /// This choice simulates the result of a standard 6-sided die roll: i.e., there are always at most 6 destinations,
        /// regardless of the size of the board. If next has a snake or ladder, you must move to the destination of that 
        /// snake or ladder.Otherwise, you move to next. The game ends when you reach the square n2. A board square on row 
        /// r and column c has a snake or ladder if board[r][c] != -1.The destination of that snake or ladder is board[r][c].
        /// Squares 1 and n2 are not the starting points of any snake or ladder. Note that you only take a snake or ladder 
        /// at most once per dice roll. If the destination to a snake or ladder is the start of another snake or ladder, 
        /// you do not follow the subsequent snake or ladder. For example, suppose the board is [[-1,4], [-1,3]], and on the 
        /// first move, your destination square is 2. You follow the ladder to square 3, but do not follow the subsequent ladder to 4. 
        /// Return the least number of dice rolls required to reach the square n2.If it is not possible to reach the square, return -1.
        /// </summary>
        /// <param name="board"></param>
        /// <returns></returns>
        public int SnakesAndLadders(int[][] board)
        {
            int bLength = board.Length;
            //   Reverse the board to traverse from top left
            Array.Reverse(board);
            //square , move
            Queue<(int square, int moves)> queue = new();
            queue.Enqueue((1, 0));
            HashSet<int> visited = new();

            while (queue.Any())
            {

                var currItem = queue.Dequeue();
                for (int i = 1; i < 7; i++)
                {
                    int nextSquare = currItem.square + i;
                    var newCoord = getInttoPos(nextSquare, bLength);
                    if (board[newCoord.row][newCoord.col] != -1)
                    {
                        nextSquare = board[newCoord.row][newCoord.col];
                    }

                    if (nextSquare == bLength * bLength)
                    {
                        return currItem.moves + 1;
                    }

                    if (!visited.Contains(nextSquare))
                    {
                        visited.Add(nextSquare);
                        queue.Enqueue((nextSquare, currItem.moves + 1));
                    }
                }
            }
            return -1;
        }

        private (int row, int col) getInttoPos(int square, int length)
        {
            int row = (square - 1) / length;
            int col = (square - 1) % length;
            if (row % 2 != 0)// if its an alternate row, traverse from right to left
            {
                //Convert the left column pointer into right and vice versa, 0th column will be come last column and last will be come 0th col.
                col = length - 1 - col;
            }
            return (row, col);
        }

        /// <summary>
        /// There are a total of numCourses courses you have to take, labeled from 0 to numCourses - 1. 
        /// You are given an array prerequisites where prerequisites[i] = [ai, bi] indicates that you 
        /// must take course bi first if you want to take course ai.
        /// For example, the pair[0, 1], indicates that to take course 0 you have to first take course 1.
        /// Return the ordering of courses you should take to finish all courses.If there are many valid 
        /// answers, return any of them.If it is impossible to finish all courses, return an empty array.
        /// </summary>
        /// <param name="numCourses"></param>
        /// <param name="prerequisites"></param>
        /// <returns></returns>
        public int[] FindOrder(int numCourses, int[][] prerequisites)
        {
            var degree = new int[numCourses];

            var parentToChildren = prerequisites.ToLookup(
                    p => p[1],
                    c => { degree[c[0]]++; return c[0]; });

            var bfs = new List<int>(numCourses);

            for (int i = 0; i < numCourses; ++i)
                if (degree[i] == 0) bfs.Add(i);

            for (int i = 0; i < bfs.Count; ++i)
            {
                foreach (var j in parentToChildren[bfs[i]])
                {
                    if (--degree[j] == 0)
                        bfs.Add(j);
                }
            }

            return bfs.Count == numCourses ? bfs.ToArray() : new int[0];
        }

        /// <summary>
        /// There are a total of numCourses courses you have to take, labeled from 0 to numCourses - 1. 
        /// You are given an array prerequisites where prerequisites[i] = [ai, bi] indicates that you 
        /// must take course bi first if you want to take course ai. For example, the pair[0, 1], indicates 
        /// that to take course 0 you have to first take course 1. Return true if you can finish all courses.
        /// Otherwise, return false.
        /// </summary>
        /// <param name="numCourses"></param>
        /// <param name="prerequisites"></param>
        /// <returns></returns>
        public bool CanFinish(int numCourses, int[][] prerequisites)
        {
            // Create adjacency list to represent the course dependencies
            List<List<int>> adjList = new List<List<int>>();
            for (int i = 0; i < numCourses; i++)
            {
                adjList.Add(new List<int>());
            }

            // Populate adjacency list with prerequisites
            foreach (int[] prerequisite in prerequisites)
            {
                int course = prerequisite[0];
                int prerequisiteCourse = prerequisite[1];
                adjList[course].Add(prerequisiteCourse);
            }

            // Create visited and recursion stack arrays
            bool[] visited = new bool[numCourses];
            bool[] recursionStack = new bool[numCourses];

            // Check for cycle in each course using DFS
            for (int course = 0; course < numCourses; course++)
            {
                if (HasCycle(course, adjList, visited, recursionStack))
                {
                    return false; // Cycle found, cannot finish all courses
                }
            }

            return true; // No cycle found, can finish all courses
        }

        private bool HasCycle(int course, List<List<int>> adjList, bool[] visited, bool[] recursionStack)
        {
            // Mark the current course as visited and add it to the recursion stack
            visited[course] = true;
            recursionStack[course] = true;

            // Traverse the prerequisites of the current course
            foreach (int prerequisiteCourse in adjList[course])
            {
                // If the prerequisite course is in the recursion stack, cycle is found
                if (recursionStack[prerequisiteCourse])
                {
                    return true;
                }

                // If the prerequisite course is not visited, recursively check for cycle
                if (!visited[prerequisiteCourse] && HasCycle(prerequisiteCourse, adjList, visited, recursionStack))
                {
                    return true;
                }
            }

            // Remove the current course from the recursion stack
            recursionStack[course] = false;

            return false;
        }

        /// <summary>
        /// You are given an array of variable pairs equations and an array of real numbers values, where equations[i] 
        /// = [Ai, Bi] and values[i] represent the equation Ai / Bi = values[i]. Each Ai or Bi is a string that represents
        /// a single variable.
        /// You are also given some queries, where queries[j] = [Cj, Dj] represents the jth query where you must find the 
        /// answer for Cj / Dj = ?.
        /// Return the answers to all queries.If a single answer cannot be determined, return -1.0. Note: The input is always 
        /// valid. You may assume that evaluating the queries will not result in division by zero and that there is no contradiction.
        /// Note: The variables that do not occur in the list of equations are undefined, so the answer cannot be determined for them.
        /// </summary>
        /// <param name="equations"></param>
        /// <param name="values"></param>
        /// <param name="queries"></param>
        /// <returns></returns>
        public double[] CalcEquation(IList<IList<string>> equations, double[] values, IList<IList<string>> queries)
        {

            HashSet<string> vis = new HashSet<string>();
            Dictionary<string, Dictionary<string, double>> d = new Dictionary<string, Dictionary<string, double>>();

            for (int i = 0; i < equations.Count; i++)
            {
                string numerator = equations[i][0];
                string denominator = equations[i][1];
                double resultValue = values[i];

                if (!d.ContainsKey(numerator))
                    d[numerator] = new Dictionary<string, double>();

                if (!d.ContainsKey(denominator))
                    d[denominator] = new Dictionary<string, double>();

                d[numerator][denominator] = resultValue;
                d[denominator][numerator] = 1 / resultValue;
            }

            return queries.Select(q => EvaluateQuery(q[0], q[1], d, vis)).ToArray();
        }

        private double EvaluateQuery(string num, string den, Dictionary<string, Dictionary<string, double>> d, HashSet<string> vis)
        {
            if (!d.ContainsKey(num) || !d.ContainsKey(den))
                return -1;

            if (num == den)
                return 1;

            if (d.ContainsKey(num) && d[num].ContainsKey(den))
                return d[num][den];

            vis.Add(num);
            double cur = -1;
            foreach (var key in d[num].Keys)
            {
                if (!vis.Contains(key))
                {
                    cur = EvaluateQuery(key, den, d, vis);
                    if (cur != -1)
                    {
                        cur = cur * d[num][key];
                        break;
                    }
                }
            }

            vis.Remove(num);
            return cur;
        }

        public Node3 CloneGraph(Node3 node)
        {
            if (node is null) return null;
            var queue = new Queue<Node3>();
            var visited = new Dictionary<Node3, (Node3, bool)>();
            queue.Enqueue(node);
            visited.Add(node, (new Node3(node.val), false));

            while (queue.Any())
            {
                var current = queue.Dequeue();
                var copy = visited[current].Item1;

                foreach (var neighbor in current.neighbors)
                {
                    if (!visited.ContainsKey(neighbor))
                        visited.Add(neighbor, (new Node3(neighbor.val), false));

                    copy.neighbors.Add(visited[neighbor].Item1);

                    if (!visited[neighbor].Item2 && !queue.Contains(neighbor))
                        queue.Enqueue(neighbor);
                }

                visited[current] = (visited[current].Item1, true);
            }

            return visited[node].Item1;
        }

        /// <summary>
        /// You are given an m x n matrix board containing letters 'X' and 'O', capture regions that are surrounded:
        /// Connect: A cell is connected to adjacent cells horizontally or vertically.
        /// Region: To form a region connect every 'O' cell.
        /// Surround: The region is surrounded with 'X' cells if you can connect the region with 'X' cells and none 
        /// of the region cells are on the edge of the board. To capture a surrounded region, replace all 'O's with 
        /// 'X's in-place within the original board.You do not need to return anything.
        /// </summary>
        /// <param name="board"></param>
        public void Solve(char[][] board)
        {
            int m = board.Length;
            int n = board[0].Length;

            // Mark all border-connected 'O's as 'T'
            for (int i = 0; i < m; i++)
            {
                if (board[i][0] == 'O') DFS(board, i, 0);
                if (board[i][n - 1] == 'O') DFS(board, i, n - 1);
            }
            for (int j = 0; j < n; j++)
            {
                if (board[0][j] == 'O') DFS(board, 0, j);
                if (board[m - 1][j] == 'O') DFS(board, m - 1, j);
            }

            // Flip all 'O's to 'X' and 'T's back to 'O'
            for (int i = 0; i < m; i++)
            {
                for (int j = 0; j < n; j++)
                {
                    board[i][j] = board[i][j] != 'T' ? 'X' : 'O';
                }
            }
        }

        private void DFS(char[][] board, int i, int j)
        {
            int m = board.Length;
            int n = board[0].Length;

            // Check if the cell is within bounds and is 'O'
            if (i < 0 || i >= m || j < 0 || j >= n || board[i][j] != 'O') return;

            // Mark this cell as 'T'
            board[i][j] = 'T';

            // Recursively mark all connected 'O's
            DFS(board, i + 1, j);
            DFS(board, i - 1, j);
            DFS(board, i, j + 1);
            DFS(board, i, j - 1);
        }

        /// <summary>
        /// Given an m x n 2D binary grid grid which represents a map of '1's (land) and '0's (water),
        /// return the number of islands. An island is surrounded by water and is formed by connecting 
        /// adjacent lands horizontally or vertically.You may assume all four edges of the grid are 
        /// all surrounded by water.
        /// </summary>
        /// <param name="grid"></param>
        /// <returns></returns>
        public int NumIslands(char[][] grid)
        {
            nr = grid.Length;
            nc = grid[0].Length;

            int totalIsland = 0;

            for (int i = 0; i < nr; i++)
            {
                for (int j = 0; j < nc; j++)
                {
                    if (grid[i][j] == '1')
                    {
                        totalIsland++;
                        findIsland(grid, i, j);
                    }
                }
            }
            return totalIsland;
        }
        public void findIsland(char[][] grid, int i, int j)
        {
            if (isValidCell(grid, i, j))
            {
                grid[i][j] = '0';
                findIsland(grid, i + 1, j);
                findIsland(grid, i - 1, j);
                findIsland(grid, i, j + 1);
                findIsland(grid, i, j - 1);
            }
        }
        private bool isValidCell(char[][] grid, int i, int j)
        {
            return i >= 0 && i < nr && j >= 0 && j < nc && grid[i][j] == '1';
        }

        /// <summary>
        /// Given an integer array nums where every element appears three times except for one,
        /// which appears exactly once. Find the single element and return it.
        /// You must implement a solution with a linear runtime complexity and use only constant
        /// extra space.
        /// </summary>
        /// <param name="nums"></param>
        /// <returns></returns>
        public int SingleNumber(int[] nums)
        {
            int ones = 0; // Tracks the bits that have appeared once
            int twos = 0; // Tracks the bits that have appeared twice

            foreach (int num in nums)
            {
                ones = (ones ^ num) & ~twos;
                twos = (twos ^ num) & ~ones;
            }

            return ones;
        }

        /// <summary>
        /// You are given two integer arrays nums1 and nums2 sorted in non-decreasing order 
        /// and an integer k. Define a pair(u, v) which consists of one element from the first 
        /// array and one element from the second array. Return the k pairs (u1, v1), (u2, v2),
        /// ..., (uk, vk) with the smallest sums.
        /// </summary>
        /// <param name="root"></param>
        /// <param name="k"></param>
        /// <returns></returns>
        public static int KthSmallest(TreeNode root, int k)
        {
            return InOrder(root).Skip(k - 1).Take(1).First();

            IEnumerable<int> InOrder(TreeNode node)
            {
                if (node is not null)
                {
                    foreach (var n in InOrder(node.left))
                    {
                        yield return n;
                    }
                    // Trace.WriteLine(node.val); // Output: 1, 2, .. k
                    yield return node.val;
                    foreach (var n in InOrder(node.right))
                    {
                        yield return n;
                    }
                }
            }
        }

        /// <summary>
        /// Given the root of a binary tree, return the zigzag level order traversal of its nodes' values.
        /// (i.e., from left to right, then right to left for the next level and alternate between).
        /// </summary>
        /// <param name="root"></param>
        /// <returns></returns>
        public IList<IList<int>> ZigzagLevelOrder(TreeNode root)
        {
            var result = new List<IList<int>>();

            if (root != null)
            {
                Queue<TreeNode> queue = new Queue<TreeNode>();
                queue.Enqueue(root);
                var reverseFlag = false;
                while (queue.Count > 0)
                {
                    var count = queue.Count;
                    var lst = new List<int>();
                    while (count-- > 0)
                    {
                        TreeNode node = queue.Dequeue();

                        if (reverseFlag)
                        {
                            lst.Insert(0, node.val);
                        }
                        else
                        {
                            lst.Add(node.val);
                        }


                        if (node.left != null)
                        {
                            queue.Enqueue(node.left);
                        }
                        if (node.right != null)
                        {
                            queue.Enqueue(node.right);
                        }
                    }
                    reverseFlag = !reverseFlag;
                    result.Add(lst);
                }
            }

            return result;
        }

        /// <summary>
        /// Given the root of a binary tree, return the level order traversal of its nodes' values.
        /// (i.e., from left to right, level by level).
        /// </summary>
        /// <param name="root"></param>
        /// <returns></returns>
        public IList<IList<int>> LevelOrder(TreeNode root)
        {
            var res = new List<IList<int>>();
            if (root == null) return res;
            Queue<TreeNode> queue = new Queue<TreeNode>();

            queue.Enqueue(root);
            while (queue.Count > 0)
            {
                List<int> l = new List<int>();
                int stop = queue.Count;
                for (int i = 0; i < stop; i++)
                {
                    TreeNode node = queue.Dequeue();
                    l.Add(node.val);
                    if (node.left != null) queue.Enqueue(node.left);
                    if (node.right != null) queue.Enqueue(node.right);
                }
                res.Add(l);
            }

            return res;
        }

        /// <summary>
        /// Given the root of a binary tree, imagine yourself standing on the right side of it, 
        /// return the values of the nodes you can see ordered from top to bottom.
        /// </summary>
        /// <param name="root"></param>
        /// <returns></returns>
        public IList<int> RightSideView(TreeNode root)
        {
            RightSide = new List<int>();
            Depth = 0;
            DFS(root, 1);
            return RightSide;
        }

        private void DFS(TreeNode node, int depth)
        {
            if (node == null) return;
            else
            {
                if (depth == Depth + 1)
                {
                    RightSide.Add(node.val);
                    Depth++;
                }
                DFS(node.right, depth + 1);
                DFS(node.left, depth + 1);
            }
        }

        /// <summary>
        /// Given a binary tree, find the lowest common ancestor (LCA) of two given nodes in the tree.
        /// According to the definition of LCA on Wikipedia: “The lowest common ancestor is defined 
        /// between two nodes p and q as the lowest node in T that has both p and q as descendants
        /// (where we allow a node to be a descendant of itself).”
        /// </summary>
        /// <param name="root"></param>
        /// <param name="p"></param>
        /// <param name="q"></param>
        /// <returns></returns>
        public static TreeNode LowestCommonAncestor(TreeNode root, TreeNode p, TreeNode q)
        {
            Find(root, p, q);
            return result;
        }

        private static bool Find(TreeNode root, TreeNode p, TreeNode q)
        {
            if (root == null) return false;

            var left = Find(root.left, p, q) ? 1 : 0;
            var right = Find(root.right, p, q) ? 1 : 0;
            var self = (root == p || root == q) ? 1 : 0;

            if (left + right + self >= 2) result = root;
            return left + right + self > 0;
        }

        /// <summary>
        /// Implement the BSTIterator class that represents an iterator over the in-order traversal 
        /// of a binary search tree (BST): BSTIterator(TreeNode root) Initializes an object of the
        /// BSTIterator class. The root of the BST is given as part of the constructor.
        /// The pointer should be initialized to a non-existent number smaller than any element in the BST.
        /// boolean hasNext() Returns true if there exists a number in the traversal to the right of the pointer,
        /// otherwise returns false. int next() Moves the pointer to the right, then returns the number at the pointer.
        /// Notice that by initializing the pointer to a non-existent smallest number, the first call to next() 
        /// will return the smallest element in the BST. You may assume that next() calls will always be valid. 
        /// That is, there will be at least a next number in the in-order traversal when next() is called.
        /// </summary>
        public class BSTIterator
        {
            Queue<int> q;
            public BSTIterator(TreeNode root)
            {
                q = new Queue<int>();
                helper(root, q);
            }
            private void helper(TreeNode root, Queue<int> q)
            {
                if (root == null) return;
                helper(root.left, q);
                q.Enqueue(root.val);
                helper(root.right, q);
            }
            public int Next()
            {
                return q.Dequeue();
            }

            public bool HasNext()
            {
                return q.Count != 0;
            }
        }

        /// <summary>
        /// You are given the root of a binary tree containing digits from 0 to 9 only.
        /// Each root-to-leaf path in the tree represents a number.
        /// For example, the root-to-leaf path 1 -> 2 -> 3 represents the number 123.
        /// Return the total sum of all root-to-leaf numbers.Test cases are generated 
        /// so that the answer will fit in a 32-bit integer. A leaf node is a node with no children.
        /// </summary>
        /// <param name="root"></param>
        /// <returns></returns>
        public static int SumNumbers(TreeNode root)
        {
            if (root == null) return 0;

            int totalSum = 0;
            var nodes = new Queue<TreeNode>();
            var sums = new Queue<int>();

            nodes.Enqueue(root);
            sums.Enqueue(root.val);

            while (nodes.Count > 0)
            {
                TreeNode node = nodes.Dequeue();
                int currentSum = sums.Dequeue();

                if (node.left == null && node.right == null)
                {
                    totalSum += currentSum;
                }

                if (node.left != null)
                {
                    nodes.Enqueue(node.left);
                    sums.Enqueue(currentSum * 10 + node.left.val);
                }

                if (node.right != null)
                {
                    nodes.Enqueue(node.right);
                    sums.Enqueue(currentSum * 10 + node.right.val);
                }
            }

            return totalSum;
        }

        /// <summary>
        /// Given the root of a binary tree, flatten the tree into a "linked list":
        /// The "linked list" should use the same TreeNode class where the right child 
        /// pointer points to the next node in the list and the left child pointer is always null.
        /// The "linked list" should be in the same order as a pre-order traversal of the binary tree.
        /// </summary>
        /// <param name="root"></param>
        public static void Flatten(TreeNode root)
        {
            var stack = new Stack<TreeNode>();
            stack.Push(root);

            TreeNode previous = null;

            while (stack.Count > 0)
            {
                var node = stack.Pop();

                if (node != null)
                {
                    if (node.right != null)
                    {
                        stack.Push(node.right);
                    }

                    if (node.left != null)
                    {
                        stack.Push(node.left);
                    }

                    if (previous != null)
                    {
                        previous.right = node;
                        previous.left = null;
                    }
                }

                previous = node;
            }
        }

        /// <summary>
        /// Populate each next pointer to point to its next right node. 
        /// If there is no next right node, the next pointer should be set to NULL.
        /// Initially, all next pointers are set to NULL.
        /// </summary>
        /// <param name="root"></param>
        /// <returns></returns>
        public static NodeC Connect(NodeC root)
        {
            if (root == null) return root;
            Queue<NodeC> q = new Queue<NodeC>();
            q.Enqueue(root);
            while (q.Count > 0)
            {
                int count = q.Count;
                NodeC prev = null;
                for (int i = 0; i < count; i++)
                {
                    NodeC cur = q.Dequeue();
                    if (cur.right != null) q.Enqueue(cur.right);
                    if (cur.left != null) q.Enqueue(cur.left);
                    cur.next = prev;
                    prev = cur;
                }
            }
            return root;
        }

        /// <summary>
        /// Given two integer arrays inorder and postorder where inorder is the inorder traversal 
        /// of a binary tree and postorder is the postorder traversal of the same tree, construct 
        /// and return the binary tree.
        /// </summary>
        /// <param name="inorder"></param>
        /// <param name="postorder"></param>
        /// <returns></returns>
        public static TreeNode BuildTree2(int[] inorder, int[] postorder)
        {
            var n = inorder.Length;

            if (postorder.Length == 0)
            {
                return null;
            }
            var lastVal = postorder.Last();
            var root = new TreeNode(lastVal);

            var idx = new List<int>(inorder).IndexOf(lastVal);
            root.left = BuildTree2(inorder[0..idx], postorder[0..idx]);
            root.right = BuildTree2(inorder[(idx + 1)..n], postorder[idx..(n - 1)]);
            return root;
        }

        /// <summary>
        /// Given two integer arrays preorder and inorder where preorder is the preorder traversal 
        /// of a binary tree and inorder is the inorder traversal of the same tree, 
        /// construct and return the binary tree.
        /// </summary>
        /// <param name="preorder"></param>
        /// <param name="inorder"></param>
        /// <returns></returns>
        public TreeNode BuildTree(int[] preorder, int[] inorder)
        {
            i = 0;
            inorderMap = new Dictionary<int, int>();

            for (int idx = 0; idx < inorder.Length; idx++)
            {
                inorderMap[inorder[idx]] = idx;
            }

            return Helper(preorder, 0, inorder.Length - 1);
        }

        private TreeNode Helper(int[] preorder, int j, int k)
        {
            if (j > k)
            {
                return null;
            }

            int nodeVal = preorder[i++];
            TreeNode node = new TreeNode(nodeVal);
            int idx = inorderMap[nodeVal];
            node.left = Helper(preorder, j, idx - 1);
            node.right = Helper(preorder, idx + 1, k);

            return node;
        }

    /// <summary>
    /// Design a data structure that follows the constraints of a Least Recently Used (LRU) cache.
    /// Implement the LRUCache class: LRUCache(int capacity) Initialize the LRU cache with positive size capacity.
    /// int get(int key) Return the value of the key if the key exists, otherwise return -1. void put(int key, int value)
    /// Update the value of the key if the key exists.Otherwise, add the key-value pair to the cache.
    /// If the number of keys exceeds the capacity from this operation, evict the least recently used key. The functions 
    /// get and put must each run in O(1) average time complexity.
    /// </summary>
    public class LRUCache
        {

            private readonly int capacity;
            private readonly Dictionary<int, LinkedListNode<CacheItem>> cacheMap;
            private readonly LinkedList<CacheItem> cacheList;

            public LRUCache(int capacity)
            {
                this.capacity = capacity;
                cacheMap = new Dictionary<int, LinkedListNode<CacheItem>>(capacity);
                cacheList = new LinkedList<CacheItem>();
            }

            public int Get(int key)
            {
                if (cacheMap.TryGetValue(key, out var node))
                {
                    cacheList.Remove(node);
                    cacheList.AddFirst(node);
                    return node.Value.Value;
                }
                return -1;
            }

            public void Put(int key, int value)
            {
                if (cacheMap.TryGetValue(key, out var node))
                {
                    node.Value.Value = value;
                    cacheList.Remove(node);
                    cacheList.AddFirst(node);
                }
                else
                {
                    if (cacheMap.Count >= capacity)
                    {
                        var lastNode = cacheList.Last;
                        cacheMap.Remove(lastNode.Value.Key);
                        cacheList.RemoveLast();
                    }

                    var newNode = new LinkedListNode<CacheItem>(new CacheItem(key, value));
                    cacheMap.Add(key, newNode);
                    cacheList.AddFirst(newNode);
                }
            }

            private class CacheItem
            {
                public int Key { get; }
                public int Value { get; set; }

                public CacheItem(int key, int value)
                {
                    Key = key;
                    Value = value;
                }
            }
        }

        /// <summary>
        /// Given the head of a linked list and a value x, partition it such 
        /// that all nodes less than x come before nodes greater than or equal to x.
        /// You should preserve the original relative order of the nodes in each of
        /// the two partitions.
        /// </summary>
        /// <param name="head"></param>
        /// <param name="x"></param>
        /// <returns></returns>
        public static ListNode Partition(ListNode head, int x)
        {
            if (head == null)
                return new ListNode();

            var dummy = new ListNode(0);
            var front = dummy;
            var dummy2 = new ListNode(0);
            var back = dummy2;
            while (head != null)
            {
                if (head.val < x)
                {
                    front.next = head;
                    front = front.next;
                }
                else
                {
                    back.next = head;
                    back = back.next;
                }
                head = head.next;
            }
            back.next = null;
            front.next = dummy2.next;
            return dummy.next;
        }

        /// <summary>
        /// Given the head of a linked list, rotate the list to the right by k places.
        /// </summary>
        /// <param name="head"></param>
        /// <param name="k"></param>
        /// <returns></returns>
        public static ListNode RotateRight(ListNode head, int k)
        {
            if (head == null)
                return new ListNode();

            var length = 1;
            var tail = head;

            while (tail.next != null)
            {
                length++;
                tail = tail.next;
            }

            tail.next = head;

            k = length - k % length;

            for (var i = 0; i < k; i++)
            {
                head = head.next;
                tail = tail.next;
            }

            tail.next = null;

            return head;
        }

        /// <summary>
        /// Given the head of a sorted linked list, delete all nodes that have duplicate numbers,
        /// leaving only distinct numbers from the original list. 
        /// Return the linked list sorted as well.
        /// </summary>
        /// <param name="head"></param>
        /// <returns></returns>
        public static ListNode DeleteDuplicates(ListNode head)
        {
            var dummyHead = new ListNode(0, head);
            var prev = dummyHead;

            while (prev != null)
            {
                // Found value that has duplicates
                if (prev.next != null && prev.next.next != null && prev.next.val == prev.next.next.val)
                {
                    var duplicateValue = prev.next.val;
                    while (prev.next != null && prev.next.val == duplicateValue) prev.next = prev.next.next;
                }
                else prev = prev.next;
            }

            return dummyHead.next;
        }

        /// <summary>
        /// Given the head of a linked list, remove the nth node from the end of the list
        /// and return its head.
        /// </summary>
        /// <param name="head"></param>
        /// <param name="n"></param>
        /// <returns></returns>
        public static ListNode RemoveNthFromEnd(ListNode head, int n)
        {
            int length = 0;
            ListNode curr = head;

            // Find the length of the linked list
            while (curr != null)
            {
                length++;
                curr = curr.next;
            }

            int traverseTill = length - n - 1;
            curr = head;

            // Traverse to the node before the one to be removed
            for (int i = 0; i < traverseTill; i++)
            {
                curr = curr.next;
            }

            // Remove the nth node from the end
            if (traverseTill == -1)
            {
                return head.next;
            }
            else
            {
                curr.next = curr.next.next;
                return head;
            }
        }

        /// <summary>
        /// Reverse K group.
        /// </summary>
        /// <param name="head"></param>
        /// <param name="k"></param>
        /// <returns></returns>
        public static ListNode ReverseKGroup(ListNode head, int k)
        {
            if (head == null || k == 1)
            {
                return head;
            }

            // Check if there are at least k nodes remaining in the list
            int count = 0;
            ListNode current = head;
            while (current != null && count < k)
            {
                current = current.next;
                count++;
            }

            if (count < k)
            {
                return head; // Not enough nodes to reverse
            }

            // Reverse the first k nodes in the current group
            ListNode prev = null;
            ListNode next = null;
            current = head;
            for (int i = 0; i < k; i++)
            {
                next = current.next;
                current.next = prev;
                prev = current;
                current = next;
            }

            // Recursively reverse the remaining part of the list
            head.next = ReverseKGroup(current, k);

            return prev; // 'prev' is now the new head of this group
        }

        /// <summary>
        /// Given the head of a singly linked list and two integers left and right where left <= right, 
        /// reverse the nodes of the list from position left to position right, and return the reversed list.
        /// </summary>
        /// <param name="head"></param>
        /// <param name="left"></param>
        /// <param name="right"></param>
        /// <returns></returns>
        public static ListNode ReverseBetween(ListNode head, int left, int right)
        {
            if (head == null || left == right)
                return head;

            ListNode dummy = new ListNode(0);
            dummy.next = head;
            ListNode prev = dummy;

            // Move prev to the node just before the left position
            for (int i = 1; i < left; i++)
            {
                prev = prev.next;
            }

            ListNode current = prev.next;
            ListNode next = null;

            // Reverse the nodes from left to right
            for (int i = left; i < right; i++)
            {
                next = current.next;
                current.next = next.next;
                next.next = prev.next;
                prev.next = next;
            }

            return dummy.next;
        }

        /// <summary>
        /// A linked list of length n is given such that each node contains an additional random pointer,
        /// which could point to any node in the list, or null.
        /// Construct a deep copy of the list.The deep copy should consist of exactly n brand new nodes, 
        /// where each new node has its value set to the value of its corresponding original node.
        /// Both the next and random pointer of the new nodes should point to new nodes in the copied list
        /// such that the pointers in the original list and copied list represent the same list state.
        /// None of the pointers in the new list should point to nodes in the original list.
        /// For example, if there are two nodes X and Y in the original list, where X.random --> Y, 
        /// then for the corresponding two nodes x and y in the copied list, x.random --> y.
        /// </summary>
        /// <param name="head"></param>
        /// <returns></returns>
        public static Node CopyRandomList(Node head)
        {
            if (head == null) return new Node(0);
            Dictionary<Node, Node> oldToNew = new Dictionary<Node, Node>();
            Node curr = head;
            while (curr != null)
            {
                oldToNew[curr] = new Node(curr.val);
                curr = curr.next;
            }
            curr = head;
            while (curr != null)
            {
                oldToNew[curr].next = curr.next != null ? oldToNew[curr.next] : null;
                oldToNew[curr].random = curr.random != null ? oldToNew[curr.random] : null;
                curr = curr.next;
            }
            return oldToNew[head];
        }

        /// <summary>
        /// You are given two non-empty linked lists representing two non-negative integers. 
        /// The digits are stored in reverse order, and each of their nodes contains a single digit. 
        /// Add the two numbers and return the sum as a linked list. You may assume the two numbers 
        /// do not contain any leading zero, except the number 0 itself.
        /// </summary>
        /// <param name="l1"></param>
        /// <param name="l2"></param>
        /// <returns></returns>
        public static ListNode AddTwoNumbers(ListNode l1, ListNode l2)
        {
            if (l1 == null) return new ListNode();
            if (l2 == null) return new ListNode();
            var head = new ListNode();
            var pointer = head;
            int curval = 0;
            while (l1 != null || l2 != null)
            {
                curval = (l1 == null ? 0 : l1.val) + (l2 == null ? 0 : l2.val) + curval;
                pointer.next = new ListNode(curval % 10);
                pointer = pointer.next;
                curval = curval / 10;
                l1 = l1?.next;
                l2 = l2?.next;
            }
            if (curval != 0)
            {
                pointer.next = new ListNode(curval);
            }
            return head.next;
        }

        /// <summary>
        /// You are given an array of strings tokens that represents an arithmetic expression in a Reverse Polish Notation.
        /// Evaluate the expression. Return an integer that represents the value of the expression.
        /// </summary>
        /// <param name="tokens"></param>
        /// <returns></returns>
        public static int EvalRPN(string[] tokens)
        {
            if (!tokens.Any()) return 0;
            Stack<int> data = new();
            foreach (string token in tokens)
                if (int.TryParse(token, out int value))
                    data.Push(value);
                else
                    data.Push(s_Funcs[token](data.Pop(), data.Pop()));
            return data.Pop();
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
                var key = System.String.Concat(str.OrderBy(c => c));
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