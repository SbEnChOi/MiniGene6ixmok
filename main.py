import torch
import torch.nn.functional as F
import random
import copy
import time

# --- 1. 설정 및 GPU 준비 ---
BOARD_SIZE = 19
WIN_COUNT = 6
POPULATION_SIZE = 100   # 한 세대당 개체 수 (논문의 해 공간 후보)
GENERATIONS = 20        # 진화 세대 수
#DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DEVICE = torch.device("cuda")
print(f"사용 장치: {DEVICE}")
# --- 2. GPU 기반 병렬 평가 함수 (핵심) ---
class GPUEvaluator:
    def __init__(self):
        # 4가지 방향에 대한 커널(필터) 생성
        # 바둑판을 훑으면서 연속된 돌을 감지하는 '눈' 역할입니다.
      #  self.kernels = self._create_kernels().to(DEVICE)
        self.kernels = [k.to(DEVICE) for k in self._create_kernels()]    
    def _create_kernels(self):
        # 6목 패턴을 찾기 위한 1x6, 6x1, 6x6 대각선 필터
        kernels = []
        # 가로 (1x6)
        h_kernel = torch.ones((1, 1, 1, 6))
        kernels.append(h_kernel)
        
        # 세로 (6x1)
        v_kernel = torch.ones((1, 1, 6, 1))
        kernels.append(v_kernel)
        
        # 대각선 \ (6x6 Identity)
        d1_kernel = torch.eye(6).reshape(1, 1, 6, 6)
        kernels.append(d1_kernel)
        
        # 대각선 / (6x6 Flip)
        d2_kernel = torch.fliplr(torch.eye(6)).reshape(1, 1, 6, 6)
        kernels.append(d2_kernel)
        
        return kernels

    def evaluate_batch(self, boards_batch, player_stone):
        """
        논문의 핵심: 여러 보드(batch)를 GPU에서 동시에 평가
        boards_batch: (Batch_Size, 19, 19) 형태의 텐서 (1:내돌, -1:상대, 0:빈칸)
        """
        batch_size = boards_batch.shape[0]
        # Conv2d 입력을 위해 (Batch, Channel, H, W) 형태로 변환
        inputs = boards_batch.view(batch_size, 1, BOARD_SIZE, BOARD_SIZE).float()
        
        total_scores = torch.zeros(batch_size).to(DEVICE)
        
        # 4방향 커널 적용
        for i, kernel in enumerate(self.kernels):
            # padding=0: 보드 밖으로 나가는 것은 무시
            # stride=1: 한 칸씩 이동하며 검사
            # conv_result의 각 픽셀 값 = 해당 위치에서 연속된 돌의 개수와 유사
            conv_result = F.conv2d(inputs, kernel)
            
            # --- 점수 계산 로직 (고도화) ---
            # 내 돌(1)이 연속된 경우: 양수값, 상대 돌(-1)이 연속된 경우: 음수값
            
            # 1. 승리 조건 (6개 연속)
            # 커널 합이 6이면 내 승리, -6이면 상대 승리
            win_mask = (conv_result == 6).float().sum(dim=(1, 2, 3))
            lose_mask = (conv_result == -6).float().sum(dim=(1, 2, 3))
            
            # 2. 공격 기회 (5개 연속 - 6목에서는 4개만 되어도 강력함)
            attack_5 = (conv_result == 5).float().sum(dim=(1, 2, 3))
            attack_4 = (conv_result == 4).float().sum(dim=(1, 2, 3))
            
            # 3. 방어 필요 (상대가 5개, 4개 연속)
            defend_5 = (conv_result == -5).float().sum(dim=(1, 2, 3))
            defend_4 = (conv_result == -4).float().sum(dim=(1, 2, 3))

            # 점수 합산 (가중치 조절)
            total_scores += win_mask * 100000
            total_scores -= lose_mask * 200000 # 지는 건 더 크게 피해야 함
            total_scores += attack_5 * 5000
            total_scores += attack_4 * 500
            total_scores -= defend_5 * 10000   # 방어가 공격보다 우선
            total_scores -= defend_4 * 1000

        return total_scores

# --- 3. 6목 AI 엔진 (유전 알고리즘) ---
class Connect6AI:
    def __init__(self):
        self.board = torch.zeros((BOARD_SIZE, BOARD_SIZE), dtype=torch.int8).to(DEVICE)
        self.evaluator = GPUEvaluator()

    def get_valid_moves(self):
        # 빈 칸 좌표 반환 (Tensor 연산으로 고속화 가능하지만 여기선 간단히)
        cpu_board = self.board.cpu()
        moves = []
        for y in range(BOARD_SIZE):
            for x in range(BOARD_SIZE):
                if cpu_board[y][x] == 0:
                    moves.append((y, x))
        return moves

    def run_genetic_algorithm(self, my_color):
        """
        논문의 협업적 인공지능 중 '선별 탐색' 부분
        GPU를 이용해 수많은 후보(모집단)를 동시에 평가합니다.
        """
        valid_moves = self.get_valid_moves()
        if len(valid_moves) < 2: return valid_moves # 빈칸 부족

        # 1. 초기 모집단 생성 (무작위 2수 조합)
        population = []
        for _ in range(POPULATION_SIZE):
            if len(valid_moves) >= 2:
                genes = random.sample(valid_moves, 2) # [(y1, x1), (y2, x2)]
                population.append(genes)
        
        best_gene = None
        best_score = -float('inf')

        print(f"\n[AI 사고 시작] 내 돌: {'흑(1)' if my_color==1 else '백(-1)'}")
        
        # --- 세대(Generation) 반복 ---
        for gen in range(GENERATIONS):
            # 2. 배치(Batch) 생성: 현재 보드를 복사해서 각 유전자의 수를 둬봄
            # boards_batch shape: (POPULATION_SIZE, 19, 19)
            boards_batch = self.board.clone().unsqueeze(0).repeat(len(population), 1, 1)
            
            for i, moves in enumerate(population):
                (y1, x1), (y2, x2) = moves
                boards_batch[i, y1, x1] = my_color
                boards_batch[i, y2, x2] = my_color

            # 3. GPU 병렬 평가 (Parallel Evaluation)
            scores = self.evaluator.evaluate_batch(boards_batch, my_color)
            
            # 4. 결과 분석 및 시각화 (상위 랭커 추출)
            scores_cpu = scores.cpu().numpy()
            sorted_indices = scores.argsort(descending=True)
            
            current_best_idx = sorted_indices[0].item()
            current_best_score = scores_cpu[current_best_idx]
            
            if current_best_score > best_score:
                best_score = current_best_score
                best_gene = population[current_best_idx]

            # 진행 상황 출력 (로그)
            if gen % 5 == 0 or gen == GENERATIONS - 1:
                print(f" >> 세대 {gen+1}/{GENERATIONS} | 최고 점수: {int(best_score)} | 현재 후보: {population[current_best_idx]}")

            # 5. 선택 및 교차 (간략화된 유전 연산)
            # 상위 20%만 살아남아 다음 세대 생성
            top_k = int(POPULATION_SIZE * 0.2)
            survivors = [population[i] for i in sorted_indices[:top_k]]
            
            new_population = survivors[:]
            while len(new_population) < POPULATION_SIZE:
                # 무작위 부모 선택 후 교차
                p1, p2 = random.sample(survivors, 2)
                child = [p1[0], p2[1]] # p1의 첫 수 + p2의 두 번째 수
                
                # 변이 (Mutation): 일정 확률로 랜덤 위치 변경
                if random.random() < 0.1:
                    child[random.randint(0, 1)] = random.choice(valid_moves)
                
                # 중복 방지
                if child[0] == child[1]: 
                    child[1] = random.choice(valid_moves)
                    
                new_population.append(child)
            
            population = new_population

        print(f"[AI 결정 완료] 최종 선택: {best_gene} (점수: {int(best_score)})")
        return best_gene

    def update_board(self, moves, color):
        for y, x in moves:
            self.board[y][x] = color

    def print_board_status(self):
        b = self.board.cpu().numpy()
        print("\n   " + " ".join([f"{i:X}" for i in range(BOARD_SIZE)])) # 0~F, 10~...
        for y in range(BOARD_SIZE):
            row_str = f"{y:2X} " # 행 번호 (16진수)
            for x in range(BOARD_SIZE):
                if b[y][x] == 0: char = "."
                elif b[y][x] == 1: char = "●" # 흑
                else: char = "○" # 백
                row_str += f"{char} "
            print(row_str)

# --- 4. 메인 실행 루프 ---
if __name__ == "__main__":
    game = Connect6AI()
    
    # 게임 시뮬레이션
    # 흑(1)이 먼저 1수 둠 (정중앙)
    center = BOARD_SIZE // 2
    print('두고 싶은 곳의 좌표를 (행,열) 형태로 입력하세요. 예: "10 10"')
    row, col = map(int, input("흑(●) 착수 위치: ").split())
    game.update_board([(row,col)], 1)
    game.print_board_status()

    # 백(-1) 차례 (AI 구동)
    # AI는 2수를 둡니다.
    best_moves = game.run_genetic_algorithm(my_color=-1)
    # 결과 반영 및 출력
    game.update_board(best_moves, -1)
    game.print_board_status()