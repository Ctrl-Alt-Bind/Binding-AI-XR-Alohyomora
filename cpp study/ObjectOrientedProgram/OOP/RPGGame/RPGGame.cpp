#include <iostream>

using namespace std;
#include "Item.h"
#include "Inventory.h"

// [시스템] 몬스터를 잡을 때, 아이템을 떨군다.
Item* DropItem() {
    if (rand() % 2 == 0) {
        Weapon* weapon = new Weapon();
        return weapon;
    }
    else
    {
        Armor* armor = new Armor();
        return armor;
    }
}

int main()
{


    for (int i = 0; i < 100; i++)
    {
        Item* item = DropItem();
        item->PrintInfo();

        if (Inventory::GetInstance()->AddItem(item)) {
            cout << "Added Item to Inventory." << endl;
        }
        else
        {
            cout << "Fail to Add Item." << endl;
            delete item;
        }
    }

    // PK 당해서 랜덤으로 아이템 드랍
    for (int i = 0; i < 20; i++)
    {
        int randIndex = rand() % MAX_SLOT;
        Item* item = Inventory::GetInstance()->GetItemAtSlot(randIndex);
        if (item != nullptr) {
            if (Inventory::GetInstance()->RemoveItem(item)) {
                cout << "Remove Item" << endl;
            }
        }
    }

    cout << "==================" << endl;
    cout << "DropItem, 캐스팅, 자식 클래스의 타입 판단, 메서드 호출" << endl;
    srand((unsigned)time(0));

    // DropItem, 캐스팅, 자식 클래스의 타입 판단, 메서드 호출
    // 나중에는 스마트 포인터를 활용하게 될 수 있음.
    for (int i = 0; i < 10; i++)
    {
        Item* item = DropItem(); 

        item->PrintInfo();

        // item->GetDamage() -> 사용 불가능 Weapon 타입이 아님.

        // 무기라면 공격력을, 방어구라면 방어력을 가져오고 싶다면?
        // 1) 가상 함수를 사용한다?
        // Item에 GetDamage 가상함수를 넣는데 이걸 방어구가 가지고 있다. 어불성설이다.
        //         
        // 2) 아이템이 어떠한 타입인지를 지정한다.

        // 직접 넣어주는 방법 (다이나믹 캐스팅과 다름)
        ItemType itemType = item->GetItemType();
        if (itemType == IT_Weapon)
        {
            Weapon* weapon = (Weapon*)item;
            weapon->GetDamage();
        }else if (itemType == IT_Armor)
        {
            Armor* armor = (Armor*)item;
            armor->GetDefence();
        }               
    }
    cout << "==================" << endl;
}

