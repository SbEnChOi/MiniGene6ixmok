"""Simple runner to choose between the two game files in this workspace.

Usage: run this script and pick 1 or 2 to run the corresponding game's main().
This keeps the two implementations separate but provides a single entrypoint.
"""
import sys


def main():
    while True:
        print("\n=== 6목 게임 실행기 ===")
        print("1) 간단한 6목 (두 플레이어 로컬) - from sixmok_game.py")
        print("2) 6목 (사용자 vs MiniMax AI) - from sixmok_with_minimax_ai.py")
        print("q) 종료")

        choice = input("선택 (1/2/q): ").strip().lower()
        if choice == '1':
            try:
                import sixmok_game
                sixmok_game.main()
            except Exception as e:
                print(f"오류: sixmok_game 실행 중 예외 발생: {e}")

        elif choice == '2':
            try:
                import sixmok_with_minimax_ai
                sixmok_with_minimax_ai.main()
            except Exception as e:
                print(f"오류: sixmok_with_minimax_ai 실행 중 예외 발생: {e}")

        elif choice == 'q' or choice == 'quit' or choice == 'exit':
            print("종료합니다.")
            sys.exit(0)
        else:
            print("잘못된 선택입니다. 1, 2 또는 q 를 입력하세요.")


if __name__ == '__main__':
    main()
