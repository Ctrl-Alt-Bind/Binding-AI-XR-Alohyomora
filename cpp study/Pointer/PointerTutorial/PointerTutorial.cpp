#include <iostream>

using namespace std;

void Swap(int* a, int* b) // 주소
{
	int temp = *a;
	*a = *b;
	*b = temp;
}
struct StatInfo {
	int hp;
	int attack;
	int deffence;
};
StatInfo CreatePlayer() {
	StatInfo info;

	cout << "플레이어 생성" << endl;

	info.hp = 100;
	info.attack = 20;
	info.deffence = 10;

	return info;

	// 복사하는 방식으로 진행. -> 원본 데이터가 훼손돼도 문제는 없다.
	// StatInfo의 크기가 엄청 커진다면? -> 만들고 복사하는 과정이 애매.

}

void CreateMonster(StatInfo* info) { // 포인터가 아니라 StatInfo를 넘긴다? 복사해서 넘기는 방식이야. 부담스럽지?
	cout << "몬스터 생성" << endl;

	info->hp = 40;
	info->attack = 8;
	info->deffence = 4;
}
void Battle(StatInfo* playerPtr, StatInfo* monsterPtr) {
	while (true)
	{
		// 데미지 계산(몬스터)
		int damage = playerPtr->attack - monsterPtr->deffence;
		if (damage < 0) damage = 0;

		monsterPtr->hp -= damage;
		if (monsterPtr->hp < 0) monsterPtr->hp = 0;

		cout << "몬스터 HP : " << monsterPtr->hp << endl;

		if (monsterPtr->hp == 0) return;

		// 데미지 계산(플레이어)
		damage = monsterPtr->attack - playerPtr->deffence;
		if (damage < 0) damage = 0;

		playerPtr->hp -= damage;
		if (playerPtr->hp < 0) playerPtr->hp = 0;

		cout << "플레이어 HP : " << playerPtr->hp << endl;

		if (playerPtr->hp == 0) return;
	}
}
int main() {
	StatInfo player; // 스택 영역에 선언
	player = CreatePlayer();

	StatInfo monster;
	CreateMonster(&monster);

	Battle(&player, &monster);

	cout << endl; 
	cout << endl;
	// 포인터 연산
	// 주소 연산자 &
	// 산술 연산자 + -
	// 간접 연산자 *
	// 간접 멤버 연산자 ->

	// ============== 주소 연산자 & ==========================
	int numbers[100] = { 1,2,3,4,5,6 };
	int* ptr = &numbers[0]; // &numbers
	*(ptr + 3) = 666; // int형 4바이트 크기 만큼 주소를 3칸 옮김 = 0번 인덱스 + 3 = 3번 인덱스

	// *ptr == numbers[0]
	// *(ptr+3) == numbers[3]


	// =============== 간접 연산자, 간접 멤버 연산자 -> =========================
	StatInfo monster;
	monster.hp = 100;
	monster.attack = 10;
	monster.deffence = 2;

	StatInfo* ptrMonster = &monster;

	cout << (*ptrMonster).hp << endl; // cout << ptrMonster->hp << endl

}
	//void* ptr = nullptr; // 어떤 형태의 주소를 가지고 있는 바구니

	//int a = 10;
	//int b = 20;

	//cout << "a : " << a << " b : " << b << endl;
	//Swap(&a, &b);

	//int* ptra = &a;
	//*ptra = 30;

	//cout << "a : " << a << " b : " << b << endl;


//	int hp = 100;
//
//	// [타입]* [이름]
//	// 주소값을 담는 바구니, 64비트 운영체제면 이 변수는 8바이트짜리 값
//	void* ptr = &hp; // hp 변수의 주소를 넣음. 
//	int* ptrInt = &hp; // hp 변수의 주소를 넣음., ptrInt에 가면 int 값이 들어있음. 
//	// float* ptrFloat = (float*)&hp; // float* 타입으로 형 변환
//	// *ptrFloat = 3.14f;
//
//	// (*ptr) -> 주소를 타고감.
//
//	cout << *ptrInt << endl;
//	*ptrInt = 300;
//	cout << *ptrInt << endl;
//	
//	cout << endl;
//
//	AddHp(&hp, 100); // 주소값을 복사해서 전달함.
//	cout << *ptrInt << endl;	
//}