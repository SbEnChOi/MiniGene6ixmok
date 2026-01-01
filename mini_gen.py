import torch
import torch.nn.functional as F
import random
import sys

# --- 설정 ---
BOARD_SIZE = 19
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# 1. AI 엔진 (이전 단계의 Hybrid AI + GPU 가속)
class GPUEvaluator:
    def __init__(self):
        self.kernels = [k.to(DEVICE) for k in self._create_kernels()]


    def _create_kernels(self):
        kernels = []
        kernels.append(torch.ones((1, 1, 1, 6)))       # 가로
        kernels.append(torch.ones((1, 1, 6, 1)))       # 세로
        kernels.append(torch.eye(6).reshape(1, 1, 6, 6)) # 대각선 \
        kernels.append(torch.fliplr(torch.eye(6)).reshape(1, 1, 6, 6)) # 대각선 /
        return kernels

    def evaluate(self, board_tensor, my_color):
        batch_size = board_tensor.shape[0]
        inputs = board_tensor.view(batch_size, 1, BOARD_SIZE, BOARD_SIZE).float()
        total_scores = torch.zeros(batch_size).to(DEVICE)
        target = 1 if my_color == 1 else -1
        
        for kernel in self.kernels:
            conv = F.conv2d(inputs, kernel, padding=0)
            # 6목 승리 점수는 매우 크게
            total_scores += (conv == target * 6).float().sum(dim=(1,2,3)) * 10000000
            total_scores += (conv == target * 5).float().sum(dim=(1,2,3)) * 500000
            total_scores += (conv == target * 4).float().sum(dim=(1,2,3)) * 10000
            # 방어 점수
            total_scores -= (conv == -target * 6).float().sum(dim=(1,2,3)) * 20000000
            total_scores -= (conv == -target * 5).float().sum(dim=(1,2,3)) * 1000000
        return total_scores

class HybridConnect6AI:
    def __init__(self):
        self.evaluator = GPUEvaluator()
        # AI 내부용 가상 보드 (실제 게임 보드와 동기화 필요)
        self.virtual_board = None 

    def get_valid_moves(self, board_tensor):
        # 최적화: 돌이 있는 곳 주변 3칸 이내만 탐색 (Heuristic)
        # 전체 탐색은 너무 느리므로 후보군을 좁힙니다.
        cpu_board = board_tensor.cpu().squeeze()
        occupied = (cpu_board != 0).nonzero(as_tuple=False)
        
        if len(occupied) == 0: return [[9, 9]] # 첫 수라면 중앙

        candidate_mask = torch.zeros((BOARD_SIZE, BOARD_SIZE), dtype=torch.bool)
        
        for y, x in occupied:
            y_min, y_max = max(0, y-2), min(BOARD_SIZE, y+3)
            x_min, x_max = max(0, x-2), min(BOARD_SIZE, x+3)
            candidate_mask[y_min:y_max, x_min:x_max] = True
            
        # 이미 돌이 있는 곳은 제외
        candidate_mask[cpu_board != 0] = False
        
        moves = candidate_mask.nonzero(as_tuple=False).tolist()
        # 너무 많으면 랜덤 샘플링 (속도 조절용)
        if len(moves) > 40: return random.sample(moves, 40)
        return moves

    def run_genetic_search(self, start_board, my_color):
        # 간단한 GA 구현
        moves = self.get_valid_moves(start_board)
        if len(moves) < 2: return 0
        
        population = [random.sample(moves, 2) for _ in range(30)]
        best_fit = -float('inf')
        
        # 짧고 굵게 5세대만 (실시간 게임 반응속도 위해)
        for _ in range(5):
            batch = start_board.unsqueeze(0).repeat(30, 1, 1).clone()
            for i, m in enumerate(population):
                batch[i, m[0][0], m[0][1]] = my_color
                batch[i, m[1][0], m[1][1]] = my_color
            
            scores = self.evaluator.evaluate(batch, my_color)
            best_fit = max(best_fit, torch.max(scores).item())
            
            # 상위 50% 생존 및 교차
            _, idx = torch.sort(scores, descending=True)
            survivors = [population[i] for i in idx[:15]]
            population = survivors[:]
            while len(population) < 30:
                p1, p2 = random.sample(survivors, 2)
                population.append([p1[0], p2[1]])
                
        return best_fit

    def get_best_move(self, game_board_tensor, my_color):
        # 1-Depth Tree Search -> GA
        valid_moves = self.get_valid_moves(game_board_tensor)
        # 후보군 2수 조합 생성 (랜덤 20개만 평가)
        candidates = []
        for _ in range(20):
            if len(valid_moves) >= 2: candidates.append(random.sample(valid_moves, 2))
        
        best_move = None
        best_score = -float('inf')
        
        print(f"Thinking... (Evaluating {len(candidates)} candidates)")
        
        for move in candidates:
            temp_board = game_board_tensor.clone()
            temp_board[move[0][0]][move[0][1]] = my_color
            temp_board[move[1][0]][move[1][1]] = my_color
            
            # 상대방 입장에서 GA 돌려서 점수 깎기 (Minimax)
            opponent_score = self.run_genetic_search(temp_board, -my_color)
            
            # 내 점수(현재) - 상대 미래 점수
            my_score_batch = temp_board.unsqueeze(0)
            my_current_score = self.evaluator.evaluate(my_score_batch, my_color).item()
            
            final_score = my_current_score - (opponent_score * 1.5)
            
            if final_score > best_score:
                best_score = final_score
                best_move = move
                
        return best_move if best_move else candidates[0]

# =========================================================
# 2. 게임 컨트롤러 (Game Logic)
# =========================================================
class Connect6Game:
    def __init__(self):
        # 1: 흑(User), -1: 백(AI), 0: 빈칸
        self.board = torch.zeros((BOARD_SIZE, BOARD_SIZE), dtype=torch.float32).to(DEVICE)
        self.ai = HybridConnect6AI()
        self.turn_count = 1 # 1부터 시작
        self.history = []

    def print_board(self):
        b = self.board.cpu().numpy()
        print("\n" + "="*40)
        # 열 번호 출력
        header = "   " + " ".join([f"{i:2}" for i in range(BOARD_SIZE)])
        print(header)
        for y in range(BOARD_SIZE):
            row_str = f"{y:2} " 
            for x in range(BOARD_SIZE):
                if b[y][x] == 1: char = "●"  # 흑
                elif b[y][x] == -1: char = "○" # 백
                else: char = "."
                row_str += f" {char} "
            print(row_str)
        print("="*40)

    def check_win(self, color):
        # CPU에서 정확한 승리 판정 (6개 연속)
        b = self.board.cpu().numpy()
        target = 1 if color == 1 else -1
        
        # 가로, 세로, 대각선 방향
        directions = [(0,1), (1,0), (1,1), (1,-1)]
        
        for y in range(BOARD_SIZE):
            for x in range(BOARD_SIZE):
                if b[y][x] != target: continue
                
                for dy, dx in directions:
                    count = 0
                    for k in range(6):
                        ny, nx = y + k*dy, x + k*dx
                        if 0 <= ny < BOARD_SIZE and 0 <= nx < BOARD_SIZE and b[ny][nx] == target:
                            count += 1
                        else:
                            break
                    if count == 6:
                        return True
        return False

    def human_input(self, count):
        moves = []
        print(f"\n[흑(●) 차례] 돌을 {count}개 두세요. (입력예시: 7 7)")
        for i in range(count):
            while True:
                try:
                    user_in = input(f"돌 {i+1} 좌표 (행 열): ")
                    if user_in.lower() == 'gg': sys.exit()
                    r, c = map(int, user_in.split())
                    
                    if 0 <= r < BOARD_SIZE and 0 <= c < BOARD_SIZE and self.board[r][c] == 0:
                        moves.append([r, c])
                        # 중복 입력 방지를 위해 임시 마킹 (화면엔 아직 반영 X)
                        break
                    else:
                        print("잘못된 좌표입니다. 다시 입력하세요.")
                except ValueError:
                    print("숫자 두 개를 공백으로 구분해 주세요.")
        return moves

    def play(self):
        print("=== Connect6 (6목) Game Start ===")
        print("Rule: 흑은 처음에 1개, 그 후 2개씩 둡니다.")
        
        game_over = False
        current_color = 1 # 흑 선공
        
        while not game_over:
            self.print_board()
            
            # 1. 몇 개 둬야 하는지 결정
            if self.turn_count == 1:
                stones_to_place = 1 # 첫 턴은 1개
            else:
                stones_to_place = 2 # 그 외엔 2개
            
            # 2. 플레이어 행동
            moves = []
            if current_color == 1: # Human (Black)
                moves = self.human_input(stones_to_place)
            else: # AI (White)
                print(f"\n[백(○) AI 차례] 생각 중입니다...")
                if stones_to_place == 1:
                    # AI가 선공일 경우 예외처리 (보통 흑이 선공이지만)
                    moves = [[BOARD_SIZE//2, BOARD_SIZE//2]]
                else:
                    # AI 알고리즘 호출
                    moves = self.ai.get_best_move(self.board, -1)
            
            # 3. 보드 업데이트
            print(f"착수: {moves}")
            for r, c in moves:
                self.board[r][c] = 1 if current_color == 1 else -1
            
            # 4. 승리 판정
            if self.check_win(current_color):
                self.print_board()
                winner = "흑(●)" if current_color == 1 else "백(○)"
                print(f"\n 게임 종료! 승리: {winner} ")
                game_over = True
                break
            
            # 5. 턴 변경 및 진행
            current_color = -current_color # 1 <-> -1 전환
            self.turn_count += 1

if __name__ == "__main__":
    game = Connect6Game()
    game.play()