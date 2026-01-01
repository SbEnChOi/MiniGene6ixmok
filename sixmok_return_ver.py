import torch
import torch.nn.functional as F # 파이토치에 있는 함수들을 F 로 불러옴 여기서 합성곱연산을 가져올거임
import random
import copy
import time
import sys
# --- 1. 설정 및 GPU 준비 ---

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # GPU를 사용할 수 있으면 gpu 로 돌리기
BOARD_SIZE=19

# ---중요한 부분 1 바둑판을 다 돌면서 점수를 매기는 역할----
class See_borad:
    def __init__(self):
        self.kernels = [k.to(DEVICE) for k in self._create_kernels()] # 빈 커널을 만들고 이제 이 커널에 육목 패턴을 넣을 거임 예를 들어서 1 1 1 1 1 1 이면 육목임
    def _create_kernels(self):
        # 이제 여기서 패턴을 넣어줄거임 
        kernels = []
        # 가로 (1x6)
        kernels.append(torch.ones((1, 1, 1, 6))) # 1 1 1 1 1 1 one는 1로 채워진 텐서를 만듬
        # 이때 1 1 1 6 은 (출력 채널 수, 입력 채널 수, 높이, 너비) 를 의미함 즉 높이 1 가로 6인 패턴을 만드는 거임.
        # 세로 (6x1)
        kernels.append(torch.ones((1, 1, 6, 1)))
        # 대각선 \ (6x6 Identity)
        kernels.append(torch.eye(6).reshape(1, 1, 6, 6))
        # 대각선 / (6x6 Flip)
        kernels.append(torch.fliplr(torch.eye(6)).reshape(1, 1, 6, 6))
        return kernels
    

    def evaluate(self, board_tensor,my_dol_color): # my_dol_color 는 내 돌의 색깔(흑/빈칸/백) 1 or 0 or -1
        batch_size = board_tensor.shape[0] # 배치 사이즈는 보드 텐서의 첫번째 차원임 즉 몇개의 보드를 한번에 볼건지
        # 여러개의 보드의 개수를 가져옴

        # 입력값을 conv 연산을 하기 위해서 구조를 바꾸는 과정: .view 는 텐서의 모양을 바꾸는 역할을 함 왜 바꾸냐면 합성곱 연산을 하기 위해선 (배치, 채널, 높이, 너비) 형태여야 하기 때문임
        inputs = board_tensor.view(batch_size, 1, BOARD_SIZE, BOARD_SIZE).float()
        total_scores = torch.zeros(batch_size).to(DEVICE) # 배치 사이즈 만큼 0으로 된 텐서를 만듬

        # 이제 합성곱 연산을 시작 : F.conv2d(inputs,hernel)
        if my_dol_color ==1: #내 돌이 흑돌이면, 흑돌을 목표로 찾음 (당연한 소리)
            target_dol_color = 1
        else: #내 돌이 백돌이면 백돌을 목표로 찾음
            target_dol_color = -1

        for kernel in self.kernels:
            conv = F.conv2d(inputs, kernel, padding=0) 
            # #1. 만약 6목이면 가중치 100000점 부여 
            # total_scores += (conv == 6 * target_dol_color).float().sum(dim=(1,2,3)) * 1000000 # .sum(dim(1,2,3)) 은 4차원 텐서에서 1,2,3 차원을 다 더해서 삭제 해서 1차원으로 만들라는 거임 그럼 결과는 (batch_size,) 형태가 됨 이는 '각 바둑판마다 6목이 몇 개 있는 지' 개수를 의미함
            # total_scores += (conv == 5 * target_dol_color).float().sum(dim=(1,2,3)) * 100000
            # total_scores += (conv == 4 * target_dol_color).float().sum(dim=(1,2,3)) * 10000
            # total_scores += (conv == 3 * target_dol_color).float().sum(dim=(1,2,3)) * 1000  
            # total_scores += (conv == 2 * target_dol_color).float().sum(dim=(1,2,3)) * 1000
            # total_scores += (conv == 1 * target_dol_color).float().sum(dim=(1,2,3)) * 100


             # 합성곱 연산을 함 padding =0 은 보드 밖으로 나가는 것은 무시함
            # 이게 어떤 느낌이냐면 아까 만든 패턴 커널을 집적 보드에 대보며 모양이 딱 맞으면 점수를 주는 느낌임
            #만약에 111111 이면 conv 값은 6이 되고 011111 이면 5가 되는 식임
            # win_mask = (conv == 6 * target_dol_color).float().sum(dim=(1,2,3)) # 6목이 몇개 있는지 세는 거임
            # lose_mask = (conv == -6 * target_dol_color).float().sum(dim=(1,2,3)) # 상대방이 6목이 몇개 있는지 세는 거임


            # # 점수 계산 로직 이걸 잘해야지 잘 둘 수 있듬 ㅎ    
            
             # 커널 합이 6이면 내 승리, -6이면 상대 승리
            total_scores += (conv == my_dol_color * 6).float().sum(dim=(1,2,3)) * 10000000
            total_scores += (conv == my_dol_color * 5).float().sum(dim=(1,2,3)) * 500000
            total_scores += (conv == my_dol_color * 4).float().sum(dim=(1,2,3)) * 10000
            # 방어 점수
            total_scores -= (conv == -my_dol_color * 6).float().sum(dim=(1,2,3)) * 20000000
            total_scores -= (conv == -my_dol_color * 5).float().sum(dim=(1,2,3)) * 1000000
            total_scores -= (conv == -my_dol_color * 4).float().sum(dim=(1,2,3)) * 10000
        
        return total_scores           

class minigeneAI:
    def __init__(self):
        self.evaluator = See_borad() # 아까 만든 눈알을 장착

    def get_valid_modes(self, board_tensor): # 유효한 수를 찾는, 빈칸을 찾는 함수
        cpu_board = board_tensor.cpu().squeeze() # for 문 같은 경우에는 GPU 보다 CPU 에 더 적합하기 때문에 cpu 사용 그리고 현재 board_tensor 는 1,19,19 행태여서 필요한 크기는 19 19 이므로 불필요한 차원을 squeeze 로 제거
        already_set =(cpu_board !=0).nonzero(as_tuple=False) # 이미 돌이 놓여진 곳의 좌표(index)를 튜플 형태로 가져옴
        cango_board = torch.zeros((BOARD_SIZE, BOARD_SIZE), dtype=torch.bool) # 후보군, 내가 갈 수 있는 새로운 판을 만듬 처음엔 다 False 임

        #  돌이 있는 곳 주변 2/3칸 이내만 갈 수 있도록(탐색하도록) 만들기 (Heuristic)
        for y, x in already_set:
            y_min, y_max = max(0, y-2), min(BOARD_SIZE, y+3)
            x_min, x_max = max(0, x-2), min(BOARD_SIZE, x+3)
            cango_board[y_min:y_max, x_min:x_max] = True
        
    # 이미 돌이 있는 곳은 제외
        cango_board[cpu_board != 0] = False
    
        moves = cango_board.nonzero(as_tuple=False).tolist()
        #방금까지 했던 작업의 보드인 cango_board 에서 True 인 좌표들만 moves 로 가져옴 이 좌표들이 내가 갈 수 있는 후보군을 모아서 list 로 형태 변환 

        # 너무 많으면 랜덤 샘플링 (속도 조절용)
       # if len(moves) > 40: return random.sample(moves, 40)
        return moves

    def genetic_search(self, board,my_dol_color):
        moves = self.get_valid_modes(board) # 유효한 수를 찾음 아까 만든 함수를 사용해서
        if len(moves) <2: return 0 # 해의 개수가 2개 미만이면 genetic search 를 할 수 없으므로 그냥 0,0 을 반환

        # 유전 알고리즘의 초기 해 집단을 설정 논문에서는 216 개의 해를 사용했지만 나는 그렇게 까지 복잡할 수 는 없어서 그냥 30개로 하장

        #유전 알고리즘을 하기 전에 일단 구조부터 정리
        '''
        이 6목에서 유전자 & 염색체 : 유전자(0,1,2,...18) 는 바둑판의 좌표 하나를 의미함 염색체는 2개의 유전자로 이루어진 하나의 해(두 수)를 의미함
        즉 염색체 = [ (유전자y1,유전자x1), (유전자y2,유전자x2) ] 이런식임
        계체군은 이런 염색체들의 모음임 population = [염색체1, 염색체2, ..., 염색체n] (이중 리스트로 구현)
        적합도 함수는 특정 염색체가 얼마나 좋은지 점수를 평가하는 평가 가준? 좋은 염색체일수록 높은 점수를 받음
        유전 연산자는 선택, 교차, 돌연변이 등이 있음
        선택은 그냥 random 으로 고르는 데 적합도 점수가 높을 수록 더 뽑일 확률이 높아짐
        교차는 방금 선택된 두 유전자(부모)를 섞어서 새로운 유전자(자식)를 만드는 과정
        마지막으로 돌연변이는 일정 확률로 유전자를 무작위로 변경하는 과정임 이게 없으면 딥러닝에서 활성함수를 넣이 않아서 선형화되는 것과 비슷함

        1. 초기 해 집단 생성: 무작위로 30개의 해를 생성 각 해는 2개의 수로 이루어짐
        2. 적합도 평가: 각 해에 대해 두 수를 보드에 놓고 See_board 클래스를 사용하여 적합도 점수를 계산
        3. 선택: 적합도 점수가 높은 해들을 선택하여 다음 세대로 전달
        4. 교차: 선택된 해들 중에서 무작위로 짝을 지어 두 수를 교환하여 새로운 해를 생성
        5. 돌연변이: 일정 확률로 해의 두 수 중 하나를 무작위로 변경
        6. 세대 반복: 5세대 동안 2~5 단계를 반복
        7. 최종 선택: 마지막 세대에서 적합도 점수가 가장 높은1개의 해를 선택하여 두 수를 반환 
        '''

        population_size =  [random.sample(moves,2) for _ in range(60)] # 30개의 해를 만듬 각 해는 2개의 수로 이루어짐 30 --> 60
        best_fit = -float('inf') # 가장 좋은 적합도 점수를 저장할 변수 초기값은 매우 작은 값

        for _ in range(5): # 5세대만 진행
            batch = board.unsqueeze(0).repeat(len(population_size),1,1).clone() # board 텐서를 유전자의 개수만큼 복제해서 배치로 만듬 (30,19,19) 형태가 됨 30개의 유전자가 다른 바둑판에서 어떻게 되는 지 시뮬레이션하는 모습침
            for i, m in enumerate(population_size): # population_size의 반환값은 2개의 수로 이루어진 해임 m = [ (y1,x1), (y2,x2) ] poulation 에서 각각 수를 뽑아서 아까 만든 30개의 판에 놓는거임
                batch[i, m[0][0], m[0][1]] = my_dol_color # 첫번째 수를 놓음
                batch[i, m[1][0], m[1][1]] = my_dol_color # 두번째 수를 놓음

            scores = self.evaluator.evaluate(batch,my_dol_color) #아까 kernel 로 만든 돌의 평가기에 방금만든 수를 넣어서 점수를 매김
            # 선택: 이제 scores 점수가 가장 높은 해를 '선택' 30개의 보드에서의 해이니깐 return 값은 30개임 그래서 torch 형태로 감싸줘야함
            best_fit = max(best_fit, torch.max(scores).item())

            sort_scores,idx = torch.sort(scores,descending=True) # 점수를 높은 순서대로 정렬하고 인덱스를 가져옴
            survivors = [population_size[i] for i in idx[:15]] # 상위 15명만 살림

            #선택
            population_size_orgin = len(population_size)
            population_size = survivors[:]
            #이제 살아남았던 애들(부모 1, 부모 2)끼리 교배 시켜서 새로운 자식들을 만드는 과정을 할거임
            while len(population_size) < population_size_orgin:
                p1, p2 = random.sample(survivors, 2)
                population_size.append([p1[0], p2[1]])
                
        return best_fit
    
    #유전 알고리즘 끝 미맥(minimax) 알고리즘 시작

    '''
    하기전에 minimax 흐름 정리
    1. 후보(candidates)로 내가 둘 수 있는 곳들중에서 몇 군데를 무작위로 고름
    2. 그곳에 돌을 두었다고 상상해봄
    3. 평가하기
        -내가 둔 돌의 점수
        -그 턴에 상대가 둘 점수 예측
    4. 내 점수 - 상대점수 = 최종 점수 해서 최종점수가 가장 높은 곳을 선택
    '''
    def minimax_search(self, board_tensor, my_dol_color):
        valid_moves = self.get_valid_modes(board_tensor) # 유효한 수들을 받아옴(리스트로)
        #이게 1번 과정
        candidates=[]
        for _ in range(70): # 일단 20개로 테스트 20-->70 
            if len(valid_moves) >=2: #예외 처리 빈칸이 2개도 안되면 안되니깐
                candidates.append(random.sample(valid_moves,2)) # 후보군 2개를 무작위로 뽑음 이걸 * 30번
        best_move = None
        best_score = -float('inf')

        print(f" 후보 수 탐색중..(후보군 {len(candidates)}개 평가중)")


        #이게 2번 상상해보기 과정
        for move in candidates:
            temp_board = board_tensor.clone() # 가상의 보드 판을 만들어서 거기서 미맥 실험
            temp_board[move[0][0], move[0][1]] = my_dol_color # 내가 두었다고 상상
            temp_board[move[1][0], move[1][1]] = my_dol_color

            #3. 평가하기
            상대점수 = self.genetic_search(temp_board, -my_dol_color) # 상대방 입장에서 유전 알고리즘 돌려서 점수 깎기
            내보드판 = temp_board.unsqueeze(0) # 내 점수를 평가하기 위해서 차원 하나를 붙여서 evlaluate 함수에 들어갈 수 있도록함 
            내보드판현재점수 = self.evaluator.evaluate(내보드판, my_dol_color).item() #돌을 두었을 때 보드판 전체를 평가함
            # 최종점수 내기리
            # 내 점수 - 상대 미래 점수
            final_score = 내보드판현재점수 - (상대점수 * 0.5) # 이1.5 를 조절해서 공격적으로 or 방어적으로로 할 수있음 
            if final_score > best_score:
                best_score = final_score
                best_move = move
        return best_move #최종적으로 자연스러운 수 반환
    # ---------- 이제 알고리즘은 끝 ----------

    # 6목 게임을 관리함. 6목 규칙, 턴 관리
class GamePlay:
    def __init__(self):
        # 1: 흑(User), -1: 백(AI), 0: 빈칸
        self.board = torch.zeros((BOARD_SIZE, BOARD_SIZE), dtype=torch.float32).to(DEVICE)
        self.ai = minigeneAI()
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
                    moves = self.ai.minimax_search(self.board, -1)
            
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
    game = GamePlay()
    game.play()


'''
    조합을 바꾸니 괜찮아짐
    candidates=70
    상대 점수  가중치 = 1--> 0.5 
    유전자의 수를 60개로 변경 
'''