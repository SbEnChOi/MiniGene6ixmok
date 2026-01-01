
import numpy as np
import time

class SixmokEvaluator:
    """6목 게임을 위한 평가함수 클래스"""

    def __init__(self):
        # 패턴별 점수 정의
        self.pattern_scores = {
            'six_or_more': 1000,  # 6개 이상 연속
            'open_five': 500,     # 방어 불가능한 5목
            'double_four': 400,   # 4-4 복합
            'four_three': 350,    # 4-3 복합
            'blocked_five': 90,   # 막힌 5목
            'gap_five': 80,       # 중간 빈칸 있는 5목
            'open_four': 75,      # 열린 4목
            'blocked_four_1': 70,  # 한쪽 막힌 4목
            'blocked_four_2': 65,  # 양쪽 막힌 4목
            'gap_four': 60,       # 중간 빈칸 있는 4목
            'double_three': 45,   # 3-3 복합
            'open_three': 35,     # 열린 3목
            'blocked_three_1': 30, # 한쪽 막힌 3목
            'blocked_three_2': 25, # 양쪽 막힌 3목
            'gap_three': 20,      # 중간 빈칸 있는 3목
            'double_two': 12,     # 2-2 복합
            'open_two': 10,       # 열린 2목
            'blocked_two_1': 8,    # 한쪽 막힌 2목
            'blocked_two_2': 6,    # 양쪽 막힌 2목
            'gap_two': 5,         # 중간 빈칸 있는 2목
            'open_one': 2,        # 열린 1목
            'blocked_one': 1      # 막힌 1목
        }
        self.directions = [(1, 0), (0, 1), (1, 1), (1, -1)]

    def evaluate_board(self, board, player):
        """전체 바둑판을 평가하여 점수 반환"""
        my_score = self._calculate_player_score(board, player)
        opponent_score = self._calculate_player_score(board, -player)
        return my_score - opponent_score

    def _calculate_player_score(self, board, player):
        """특정 플레이어의 점수 계산"""
        total_score = 0
        checked_positions = set()

        for row in range(19):
            for col in range(19):
                if board[row][col] == player:
                    for dr, dc in self.directions:
                        pos_key = (row, col, dr, dc)
                        if pos_key not in checked_positions:
                            score = self._evaluate_line(board, row, col, dr, dc, player)
                            total_score += score
                            self._mark_line_as_checked(checked_positions, row, col, dr, dc, board, player)
        return total_score

    def _mark_line_as_checked(self, checked_positions, row, col, dr, dc, board, player):
        """연속된 돌들이 있는 라인을 모두 체크된 것으로 표시"""
        positions = [(row, col)]

        # 정방향
        r, c = row + dr, col + dc
        while 0 <= r < 19 and 0 <= c < 19 and board[r][c] == player:
            positions.append((r, c))
            r += dr
            c += dc

        # 역방향
        r, c = row - dr, col - dc
        while 0 <= r < 19 and 0 <= c < 19 and board[r][c] == player:
            positions.append((r, c))
            r -= dr
            c -= dc

        for pos_r, pos_c in positions:
            checked_positions.add((pos_r, pos_c, dr, dc))

    def _evaluate_line(self, board, row, col, dr, dc, player):
        """특정 방향의 라인을 평가하여 점수 반환"""
        line_info = self._analyze_line(board, row, col, dr, dc, player)

        consecutive = line_info['consecutive']
        open_ends = line_info['open_ends']

        if consecutive >= 6:
            return self.pattern_scores['six_or_more']
        elif consecutive == 5:
            if open_ends == 2:
                return self.pattern_scores['open_five']
            elif open_ends == 1:
                return self.pattern_scores['blocked_five']
            else:
                return self.pattern_scores['gap_five']
        elif consecutive == 4:
            if open_ends == 2:
                return self.pattern_scores['open_four']
            elif open_ends == 1:
                return self.pattern_scores['blocked_four_1']
            else:
                return self.pattern_scores['blocked_four_2']
        elif consecutive == 3:
            if open_ends == 2:
                return self.pattern_scores['open_three']
            elif open_ends == 1:
                return self.pattern_scores['blocked_three_1']
            else:
                return self.pattern_scores['blocked_three_2']
        elif consecutive == 2:
            if open_ends == 2:
                return self.pattern_scores['open_two']
            elif open_ends == 1:
                return self.pattern_scores['blocked_two_1']
            else:
                return self.pattern_scores['blocked_two_2']
        elif consecutive == 1:
            if open_ends >= 1:
                return self.pattern_scores['open_one']
            else:
                return self.pattern_scores['blocked_one']
        return 0

    def _analyze_line(self, board, row, col, dr, dc, player):
        """라인을 분석하여 연속성, 열린 끝 등을 계산"""
        consecutive = 1  # 현재 돌 포함

        # 정방향 카운트
        r, c = row + dr, col + dc
        while 0 <= r < 19 and 0 <= c < 19 and board[r][c] == player:
            consecutive += 1
            r += dr
            c += dc
        forward_end = (r, c)

        # 역방향 카운트
        r, c = row - dr, col - dc
        while 0 <= r < 19 and 0 <= c < 19 and board[r][c] == player:
            consecutive += 1
            r -= dr
            c -= dc
        backward_end = (r, c)

        # 열린 끝 계산
        open_ends = 0
        fr, fc = forward_end
        if 0 <= fr < 19 and 0 <= fc < 19 and board[fr][fc] == 0:
            open_ends += 1

        br, bc = backward_end
        if 0 <= br < 19 and 0 <= bc < 19 and board[br][bc] == 0:
            open_ends += 1

        return {'consecutive': consecutive, 'open_ends': open_ends}


class MinimaxAI:
    """MiniMax 알고리즘을 사용한 6목 AI"""

    def __init__(self, max_depth=4):
        self.max_depth = max_depth
        self.evaluator = SixmokEvaluator()
        self.nodes_evaluated = 0
        self.pruning_count = 0

    def get_best_move(self, board, player, time_limit=5.0):
        """최적의 수를 찾아 반환"""
        self.nodes_evaluated = 0
        self.pruning_count = 0
        start_time = time.time()

        print(f" 미맥이 사고 중... (깊이: {self.max_depth})")

        best_move = None
        best_score = float('-inf')

        valid_moves = self._get_valid_moves(board)
        if not valid_moves:
            return None

        valid_moves = self._sort_moves_by_heuristic(board, valid_moves, player)

        for i, (row, col) in enumerate(valid_moves):
            if time.time() - start_time > time_limit:
                print(f" 시간 제한. {i+1}/{len(valid_moves)}개 수 탐색")
                break

            new_board = self._make_move(board, row, col, player)
            score = self._minimax(new_board, self.max_depth - 1, False, player, 
                                float('-inf'), float('inf'))

            if score > best_score:
                best_score = score
                best_move = (row, col)

        elapsed_time = time.time() - start_time
        print(f"선택: {best_move} (점수: {best_score})")
        print(f"노드: {self.nodes_evaluated}, 가지치기: {self.pruning_count}")
        print(f"시간: {elapsed_time:.2f}초")

        return best_move

    def _minimax(self, board, depth, maximizing_player, original_player, alpha, beta):
        """MiniMax 알고리즘 with Alpha-Beta 가지치기"""
        self.nodes_evaluated += 1

        if depth == 0 or self._is_game_over(board):
            return self.evaluator.evaluate_board(board, original_player)

        current_player = original_player if maximizing_player else -original_player
        valid_moves = self._get_valid_moves(board)

        if not valid_moves:
            return self.evaluator.evaluate_board(board, original_player)

        valid_moves = self._sort_moves_by_heuristic(board, valid_moves, current_player)

        if maximizing_player:
            max_eval = float('-inf')
            for row, col in valid_moves:
                new_board = self._make_move(board, row, col, current_player)
                eval_score = self._minimax(new_board, depth - 1, False, original_player, alpha, beta)
                max_eval = max(max_eval, eval_score)
                alpha = max(alpha, eval_score)

                if beta <= alpha:
                    self.pruning_count += 1
                    break
            return max_eval
        else:
            min_eval = float('inf')
            for row, col in valid_moves:
                new_board = self._make_move(board, row, col, current_player)
                eval_score = self._minimax(new_board, depth - 1, True, original_player, alpha, beta)
                min_eval = min(min_eval, eval_score)
                beta = min(beta, eval_score)

                if beta <= alpha:
                    self.pruning_count += 1
                    break
            return min_eval

    def _get_valid_moves(self, board):
        """현재 놓을 수 있는 모든 좌표 반환"""
        valid_moves = []
        stone_count = np.count_nonzero(board)

        if stone_count < 10:
            # 초반: 중앙 영역
            for row in range(5, 14):
                for col in range(5, 14):
                    if board[row][col] == 0:
                        valid_moves.append((row, col))
        else:
            # 중반: 기존 돌 주변
            for row in range(19):
                for col in range(19):
                    if board[row][col] == 0 and self._has_nearby_stone(board, row, col, radius=3):
                        valid_moves.append((row, col))

        return valid_moves

    def _has_nearby_stone(self, board, row, col, radius=3):
        """주변에 돌이 있는지 확인"""
        for dr in range(-radius, radius + 1):
            for dc in range(-radius, radius + 1):
                if dr == 0 and dc == 0:
                    continue
                r, c = row + dr, col + dc
                if 0 <= r < 19 and 0 <= c < 19 and board[r][c] != 0:
                    return True
        return False

    def _sort_moves_by_heuristic(self, board, moves, player):
        """휴리스틱 점수에 따라 수를 정렬"""
        move_scores = []
        for row, col in moves:
            new_board = self._make_move(board, row, col, player)
            score = self.evaluator.evaluate_board(new_board, player)
            move_scores.append((score, row, col))

        move_scores.sort(reverse=True)
        return [(row, col) for score, row, col in move_scores]

    def _make_move(self, board, row, col, player):
        """수를 둔 새로운 보드 상태 생성"""
        new_board = board.copy()
        new_board[row][col] = player
        return new_board

    def _is_game_over(self, board):
        """게임이 종료되었는지 확인"""
        for row in range(19):
            for col in range(19):
                if board[row][col] != 0:
                    if self._check_winner_at(board, row, col):
                        return True
        return False

    def _check_winner_at(self, board, row, col):
        """특정 위치에서 6목이 완성되었는지 확인"""
        player = board[row][col]
        directions = [(1,0), (0,1), (1,1), (1,-1)]

        for dr, dc in directions:
            count = 1

            r, c = row + dr, col + dc
            while 0 <= r < 19 and 0 <= c < 19 and board[r][c] == player:
                count += 1
                r += dr
                c += dc

            r, c = row - dr, col - dc
            while 0 <= r < 19 and 0 <= c < 19 and board[r][c] == player:
                count += 1
                r -= dr
                c -= dc

            if count >= 6:
                return True
        return False


class SixmokGameWithAI:
    """MiniMax AI가 통합된 6목 게임"""

    def __init__(self, ai_depth=4):
        self.board = np.zeros((19, 19), dtype=int)
        self.current_player = 1  # 1: 흑돌(사용자), -1: 백돌(AI)
        self.game_over = False
        self.winner = None
        self.ai = MinimaxAI(max_depth=ai_depth)

    def display_board(self):
        """19x19 바둑판을 콘솔에 예쁘게 출력"""
        print("\n" + "="*60)
        print("           6목 게임 (사용자 vs MiniMax AI)")
        print("="*60)

        print("    ", end="")
        for col in range(19):
            print(f"{col:2}", end=" ")
        print()

        for row in range(19):
            print(f"{row:2}: ", end="")
            for col in range(19):
                if self.board[row][col] == 1:
                    print(" ●", end=" ")  # 흑돌 (사용자)
                elif self.board[row][col] == -1:
                    print(" ○", end=" ")  # 백돌 (AI)
                else:
                    print(" ·", end=" ")  # 빈칸
            print(f" :{row}")

        print("    ", end="")
        for col in range(19):
            print(f"{col:2}", end=" ")
        print()

        if self.current_player == 1:
            print(f"\n현재 차례: ●(사용자 - 흑돌)")
        else:
            print(f"\n현재 차례: ○(AI - 백돌)")
        print("-"*60)

    def is_valid_move(self, row, col):
        """유효한 착수인지 확인"""
        if not (0 <= row < 19 and 0 <= col < 19):
            return False, "좌표가 범위를 벗어났습니다."
        if self.board[row][col] != 0:
            return False, "이미 돌이 놓여진 자리입니다."
        return True, "유효한 착수입니다."

    def place_stone(self, row, col):
        """돌을 놓고 게임 상태 업데이트"""
        valid, message = self.is_valid_move(row, col)
        if not valid:
            return False, message

        self.board[row][col] = self.current_player

        if self.check_winner_at(row, col):
            self.game_over = True
            self.winner = self.current_player
            winner_name = "사용자(●)" if self.current_player == 1 else "AI(○)"
            return True, f" 게임 종료! {winner_name}이 승리했습니다!"

        if np.count_nonzero(self.board) == 361:
            self.game_over = True
            self.winner = 0
            return True, " 무승부입니다!"

        self.current_player *= -1
        return True, "착수 완료"

    def check_winner_at(self, row, col):
        """특정 위치에서 6목이 완성되었는지 확인"""
        player = self.board[row][col]
        directions = [(1,0), (0,1), (1,1), (1,-1)]

        for dr, dc in directions:
            count = 1

            r, c = row + dr, col + dc
            while 0 <= r < 19 and 0 <= c < 19 and self.board[r][c] == player:
                count += 1
                r += dr
                c += dc

            r, c = row - dr, col - dc
            while 0 <= r < 19 and 0 <= c < 19 and self.board[r][c] == player:
                count += 1
                r -= dr
                c -= dc

            if count >= 6:
                return True
        return False

    def get_ai_move(self):
        """AI의 다음 수 계산"""
        return self.ai.get_best_move(self.board, self.current_player, time_limit=5.0)

    def get_user_input(self):
        """사용자로부터 좌표 입력 받기"""
        while True:
            try:
                user_input = input("착수할 좌표 (행,열 예: 9,9) 또는 'quit': ").strip()

                if user_input.lower() == 'quit':
                    return None, None

                coords = user_input.split('')
                if len(coords) != 2:
                    print("'y x' 형식으로 입력해주세요.")
                    continue

                row = int(coords[0].strip())
                col = int(coords[1].strip())
                return row, col

            except ValueError:
                print(" 숫자를 입력해주세요.")
            except KeyboardInterrupt:
                print("\n게임을 종료합니다.")
                return None, None


def main():
    """메인 게임 실행"""
    print("6목 게임 with MiniMax AI")
    print("="*50)
    print("규칙: 6개 이상 연속된 돌을 만들면 승리")
    print("사용자(●) vs AI(○)")
    print("="*50)

    # 난이도 선택
    while True:
        try:
            difficulty = input("AI 난이도 (1:쉬움, 2:보통, 3:어려움, 4:최고): ").strip()
            if difficulty in ['1', '2', '3', '4']:
                ai_depth = int(difficulty)
                break
            else:
                print(" 1-4 사이의 숫자를 입력해주세요.")
        except:
            print("올바른 숫자를 입력해주세요.")

    game = SixmokGameWithAI(ai_depth=ai_depth)

    while not game.game_over:
        game.display_board()

        if game.current_player == 1:  # 사용자 턴
            row, col = game.get_user_input()

            if row is None:
                print(" 게임을 종료합니다.")
                return

            success, message = game.place_stone(row, col)
            if success:
                print(f"{message}")
            else:
                print(f"{message}")
                continue

        else:  # AI 턴
            ai_move = game.get_ai_move()

            if ai_move is None:
                print("AI가 수를 찾지 못했습니다.")
                break

            row, col = ai_move
            success, message = game.place_stone(row, col)
            print(f" AI 착수: ({row}, {col})")
            print(f" {message}")

    game.display_board()

    if game.winner == 1:
        print(" 인권을 지켜냄 ")
    elif game.winner == -1:
        print(" AI 승리, 인간시대의 끝이 도래함 ")
    else:
        print("무승부")


if __name__ == "__main__":
    main()
