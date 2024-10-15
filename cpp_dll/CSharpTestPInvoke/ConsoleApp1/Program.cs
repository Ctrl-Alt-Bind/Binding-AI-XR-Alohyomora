using System;
using System.Runtime.InteropServices;

namespace ConsoleApp1
{
    internal class Program
    {
        // C++ DLL에서 Generate RandomVector 함수 호출(포인터 반환)
        [DllImport("testDll1.dll", CallingConvention = CallingConvention.Cdecl)]
        public static extern IntPtr GenerateRandomVector(int size, ref int outsize);

        [DllImport("testDll1.dll", CallingConvention = CallingConvention.Cdecl)]
        public static extern double ProcessDataAndReturnTime(int dataNum);

        static void Main(string[] args)
        {
            GetDataProcessingTime(1000000);

            GetRandomVectorFromDll();
        }

        static void GetRandomVectorFromDll()
        {
            int size = 10;  // 요청할 벡터 크기
            int actualSize = 0; // c++에서 반환된 벡터 크기

            // C++에서 데이터 포인터 가져오기
            IntPtr ptr = GenerateRandomVector(size, ref actualSize);

            if(actualSize > 0)
            {
                // 포인터를 C# 배열로 변환
                int[] result = new int[actualSize];
                Marshal.Copy(ptr, result, 0, actualSize);

                Console.WriteLine("Random Vector Data:");
                foreach (int value in result)
                {
                    Console.Write(value + " ");
                }
                Console.WriteLine();
                Console.WriteLine();
            }            
        }
        static void GetDataProcessingTime(int dataNum)
        {
            double timeTakenCpp = ProcessDataAndReturnTime(1000000);

            Console.WriteLine($"Data Processing Time [C++]: {timeTakenCpp} seconds.");

            // C# 처리 속도 체크
            DateTime startTime = DateTime.Now;
            for (int i = 0; i < dataNum; i++) { };
            DateTime endTime = DateTime.Now;
            TimeSpan elapsedTime = endTime - startTime;

            Console.WriteLine($"Data Processing Time [C#]: {elapsedTime.TotalSeconds} seconds.");

        }
    }
}
