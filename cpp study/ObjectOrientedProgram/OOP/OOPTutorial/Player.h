#pragma once
#include <iostream>
using namespace std;

// ========= 클래스 [설계도] 만들기 =========  
class Player {
public:
	// 기본 생성자(constructor) 멤버 변수 초기화해라~
	Player() {
		cout << "Player 생성" << endl;
		m_hp = 0;
		m_attack = 0;
		m_defence = 0;
	};

	// 복사 생성자
	Player(const Player& other) {
		cout << "Player 생성(복사 생성)" << endl;
		this->m_hp = other.m_hp;
		this->m_attack = other.m_attack;
		this->m_defence = other.m_defence;		
	}

	// 기타 생성자
	Player(int hp, int attack, int defence) {
		// 멤버 변수가 반드시 들어가 있어야 한다면 매개변수로 값을 넣어 초기화 해야 한다.

		cout << "Player 생성(멤버 변수 초기화)" << endl;
		m_hp = hp;
		m_attack = attack;
		this->m_defence = defence;
	}

	// 소멸자(destructor)
	~Player() {
		cout << "Player 소멸" << endl;
	};
	// 멤버 함수 ->  함수, 메서드는 메모리를 다 먹지 않는다. knight 1000마리 생성 -> 1000배의 메모리 생성하는 게 아님.
	void Attack();
	void Die();
	void Heal(int value);

public:
	// 멤버 변수 (일반 변수와 다름)
	int m_hp; // _, m_, m -> 이 멤버 변수다라고 알려주는 네이밍 규칙 팀 별로 다름.
	int m_attack;
	int m_defence;

};

