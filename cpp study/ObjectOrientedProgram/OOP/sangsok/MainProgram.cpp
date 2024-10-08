#include <iostream>

using namespace std;


// OOP 3대 요소
// - 상속성 << inheritance (물려받는다..)
// - 은닉성 << data hiding (Encapsulation)
// - 다형성

// 설계도

// Is-A vs Has-A
// Is-A 상속성

class Player {
public:
	void Move() {}
	void Attack() {}
	void Die() {}

public:
	// PlayerType m_type;
	int m_hp;
	int m_attack;
	int m_defence;
};
enum PLAYER_TYPE {
	Knight = 1,
	Archer = 2,
	Magician = 3,
};
class Knight : public Player{ 
public:

public:
};

class Archer : public Player{
public:

public:
	int m_arrowCount;
};

class Magician : public Player{
public:
public:
	int m_mp;
};

void Fight(Player* p1, Player* p2) {


}


int main() {
	// 캐스팅(타입 변환)
	
	Knight k1;
	Player* p1 = &k1;

	Magician m1;
	Player* p2 = &m1;

}