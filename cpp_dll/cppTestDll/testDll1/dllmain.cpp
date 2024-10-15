// dllmain.cpp : DLL 애플리케이션의 진입점을 정의합니다.
#include "pch.h"
#include <chrono>

#include <vector>
#include <cstdlib>
#include <ctime>

BOOL APIENTRY DllMain( HMODULE hModule,
                       DWORD  ul_reason_for_call,
                       LPVOID lpReserved
                     )
{
    switch (ul_reason_for_call)
    {
    case DLL_PROCESS_ATTACH:
    case DLL_THREAD_ATTACH:
    case DLL_THREAD_DETACH:
    case DLL_PROCESS_DETACH:
        break;
    }
    return TRUE;
}

// 데이터 처리 후 처리 시간을 반환하는 함수
extern "C" __declspec(dllexport) double ProcessDataAndReturnTime() {
    using namespace std::chrono;

    // 시작 시간
    auto start = high_resolution_clock::now();

    // 데이터 처리 (여기서는 예시로 간단한 루프 사용)
    for (volatile int i = 0; i < 1000000; ++i);

    // 종료 시간
    auto end = high_resolution_clock::now();

    // 경과 시간 계산
    duration<double> elapsed = end - start;
    return elapsed.count(); // 처리 시간을 초 단위로 반환
}

// 랜덤 데이터를 생성하여 벡터로 반환
extern "C" __declspec(dllexport) int* GenerateRandomVector(int size, int* outsize) {
    static std::vector<int> randomData;

    // 벡터 크기 설정
    randomData.reserve(size);

    // 난수 생성
    std::srand(static_cast<unsigned>(std::time(0)));
    for (int i = 0; i < size; i++)
    {
        randomData[i] = std::rand() % 100; // 0 ~ 99 사이의 난수 생성
    }
    
    // 데이터 크기 설정
    *outsize = randomData.size();
    // 벡터 데이터 포인터를 반환
    return randomData.data();
}