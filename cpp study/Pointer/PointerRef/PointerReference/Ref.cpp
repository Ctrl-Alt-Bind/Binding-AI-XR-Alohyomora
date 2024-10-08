#include <iostream>

using namespace std;

struct StatInfo
{
	int hp;
	int attack;
	int deffence;
};
 
// 1) 값(복사) 전달 방식
void PrintByCopy(StatInfo player) {
	cout << "--------------------------" << endl;
	cout << "HP : " << player.hp << endl;
	cout << "Atk : " << player.attack << endl;
	cout << "Def : " << player.deffence << endl;
	cout << "--------------------------" << endl;
}

// 2) 주소 전달 방식 -> 1) 원본을 건드리고 싶을 때, 2) 복사 비용을 아끼고 싶을 때
void PrintByPointer(StatInfo* player) {
	cout << "--------------------------" << endl;
	cout << "HP : " << (*player).hp << endl;
	cout << "Atk : " << player->attack << endl;
	cout << "Def : " << player->deffence << endl;
	cout << "--------------------------" << endl;
}

// 3) 참조 전달 방식
#define OUT // ->> OUT 원본 데이터를 수정하겠다~ 실제로 OUT 뒤가 비어 있어서 의미가 없다.

void PrintByRef(OUT StatInfo& player) { // void PrintByRef(const StatInfo& player) -> 내부 데이터 수정하지 않고 복사 (비용을 아끼기 위해 참조 사용)
	cout << "--------------------------" << endl;
	cout << "HP : " << player.hp << endl;
	cout << "Atk : " << player.attack << endl;
	cout << "Def : " << player.deffence << endl;
	cout << "--------------------------" << endl;
}

int main() {
	StatInfo player = { 100, 10, 1 };
	
	PrintByCopy(player);
	PrintByPointer(&player);

	StatInfo* ptr = nullptr;


	StatInfo* info = &player;
	
	StatInfo& ref = player; // 주소 값을 이용해서 원본을 건드린다. player에 ref라는 별칭을 붙인다.
	// 참조는 내부 원리는 포인터지만 사용은 일반 변수처럼 쓴다.
	// ref.hp = 10;
	PrintByRef(OUT player);
}