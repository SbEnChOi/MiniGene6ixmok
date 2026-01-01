
import numpy as np

class SixmokGame:
    def __init__(self):
        self.board = np.zeros((19, 19), dtype=int)  # 0: 빈칸, 1: 흑돌(O), -1: 백돌(X)
        self.current_player = 1  # 1: 흑돌 먼저 시작
        self.game_over = False
        self.winner = None

    def display_board(self):
        """19x19 바둑판을 콘솔에 예쁘게 출력"""
        print("\n" + "="*60)
        print("                    6목(19x19)")
        print("="*60)

        # 열 번호 표시 (0-18)
        print("    ", end="")
        for col in range(19):
            print(f"{col:2}", end=" ")
        print()

        # 행과 바둑판 내용 출력
        for row in range(19):
            print(f"{row:2}: ", end="")
            for col in range(19):
                if self.board[row][col] == 1:
                    print(" O", end=" ")  # 흑돌
                elif self.board[row][col] == -1:
                    print(" X", end=" ")  # 백돌
                else:
                    print(" ·", end=" ")  # 빈칸
            print(f" :{row}")

        # 하단 열 번호 다시 표시
        print("    ", end="")
        for col in range(19):
            print(f"{col:2}", end=" ")
        print()

        # 현재 플레이어 표시
        player_symbol = "O(흑돌)" if self.current_player == 1 else "X(백돌)"
        print(f"\n현재 차례: {player_symbol}")
        print("-"*60)

    def is_valid_move(self, row, col):
        """유효한 착수인지 확인"""
        if not (0 <= row < 19 and 0 <= col < 19):
            return False, "좌표가 바둑판 범위를 벗어났습니다."
        if self.board[row][col] != 0:
            return False, "이미 돌이 놓여진 자리입니다."
        return True, "유효한 착수입니다."

    def place_stone(self, row, col):
        """돌을 놓고 게임 상태 업데이트"""
        valid, message = self.is_valid_move(row, col)
        if not valid:
            return False, message

        self.board[row][col] = self.current_player

        # 승리 조건 확인
        if self.check_winner_at(row, col):
            self.game_over = True
            self.winner = self.current_player
            return True, f" 게임 종료! {('흑돌(O)' if self.current_player == 1 else '백돌(X)')}이 승리했습니다!"

        # 플레이어 교체
        self.current_player *= -1
        return True, "착수 완료"

    def check_winner_at(self, row, col):
        """특정 위치에서 6목이 완성되었는지 확인"""
        player = self.board[row][col]
        directions = [(1,0), (0,1), (1,1), (1,-1)]  # 가로, 세로, 대각선

        for dr, dc in directions:
            count = 1  # 현재 돌 포함

            # 한 방향으로 카운트
            r, c = row + dr, col + dc
            while 0 <= r < 19 and 0 <= c < 19 and self.board[r][c] == player:
                count += 1
                r += dr
                c += dc

            # 반대 방향으로 카운트
            r, c = row - dr, col - dc
            while 0 <= r < 19 and 0 <= c < 19 and self.board[r][c] == player:
                count += 1
                r -= dr
                c -= dc

            if count >= 6:  # 6목 달성
                return True
        return False

    def get_user_input(self):
        """사용자로부터 좌표 입력 받기"""
        while True:
            try:
                user_input = input("착수할 좌표를 입력하세요 (행,열 형식 예: 9,9) 또는 'quit' 종료: ").strip()

                if user_input.lower() == 'q':
                    return None, None

                # 띄어쓰기로 구분 
                coords = user_input.split(' ')
                # if len(coords) != 1:
                #     print("잘못된 형식입니다.형식으로 입력해주세요. 예: 9 9")
                #     continue

                row = int(coords[0].strip())
                col = int(coords[1].strip())

                return row, col

            except ValueError:
                print("숫자를 입력해주세요. 예: 9,9")
            except KeyboardInterrupt:
                print("\n게임을 종료합니다.")
                return None, None

def main():
    """메인 게임 루프"""
    print("6목 게임에 오신 것을 환영합니다!")
    print("규칙: 6개 이상의 연속된 돌을 만들면 승리합니다.")
    print("좌표 입력 방법: 행 열 (예: 9 9)")
    print("게임 종료: 'q' 입력")

    game = SixmokGame()

    while not game.game_over:
        game.display_board()

        # 사용자 입력 받기
        row, col = game.get_user_input()

        if row is None:  # 게임 종료
            print("게임 종료")
            break

        # 착수 시도
        success, message = game.place_stone(row, col)

        if success:
            print(f"{message}")
            if game.game_over:
                game.display_board()
                print(f"이긴사람: {('흑돌(O)' if game.winner == 1 else '백돌(X)')}")
                break
        else:
            print(f"X {message}")

if __name__ == "__main__":
    main()
