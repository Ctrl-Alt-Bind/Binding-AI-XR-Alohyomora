#include <iostream>

using namespace std;

#include "Player.h"

// 객체 지향
// 데이터 + 로직


int main() {

	// 객체 각각은 생성된 이후 별개로 동작한다.
	Player k1; // 스택 영역에 객체 생성
	k1.m_hp = 200;
	k1.Attack();
	Player k2(100, 10, 1); // 기타 생성자 이용

	Player k3(k2); // 복사 생성자 이용
}