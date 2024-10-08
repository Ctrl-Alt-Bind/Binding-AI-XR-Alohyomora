#include <iostream>

using namespace std;

// 프레임이 터졌다. 99%는 NULL 크래시
// 정수 오버플로우(언더플로우)
// Memory Leak 메모리 누수 : 메모리 할당 해제
// 메모리 오염
// - Use-After-Free (참조하고 있는 주소값이 삭제되었는데도 불구하고, 걔를 계속 건드리기 시작하다 그게 중요한 메모리와 겹치게 되면 뿅 터질 수 있다.
// - 버퍼 오버플로우 (int arr[100]인데 arr[101]에 접근)
// - 캐스팅 (캐스팅 과정에서 변환이 잘못 일어난 경우)

class Player {
public:
    int hp = 0;
};

Player* FindPlayer(int id) {

    return nullptr;
}
void Test(Player* p) {
    p->hp = 10;
}

int main()
{
    // NULL 체크 필수!
    Player* p = FindPlayer(0);
    Test(p);

    short value = 30000; // 최대치를 넘으면 -(음수)로 넘어감 ex) 체력 회복을 했는데 음수로 변해서 바로 죽어 버린다.

    
}


