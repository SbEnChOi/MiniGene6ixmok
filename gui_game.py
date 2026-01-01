import sys
import torch
import torch.nn.functional as F
import random
import tkinter as tk
from tkinter import messagebox
import threading

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BOARD_SIZE = 19
CELL_SIZE = 35

class See_borad:
    def __init__(self):
        self.kernels = [k.to(DEVICE) for k in self._create_kernels()]
    
    def _create_kernels(self):
        kernels = []
        kernels.append(torch.ones((1, 1, 1, 6)))
        kernels.append(torch.ones((1, 1, 6, 1)))
        kernels.append(torch.eye(6).reshape(1, 1, 6, 6))
        kernels.append(torch.fliplr(torch.eye(6)).reshape(1, 1, 6, 6))
        return kernels
    
    def evaluate(self, board_tensor, my_dol_color):
        batch_size = board_tensor.shape[0]
        inputs = board_tensor.view(batch_size, 1, BOARD_SIZE, BOARD_SIZE).float()
        total_scores = torch.zeros(batch_size).to(DEVICE)
        
        if my_dol_color == 1:
            target_dol_color = 1
        else:
            target_dol_color = -1
        
        for kernel in self.kernels:
            conv = F.conv2d(inputs, kernel, padding=0)
            total_scores += (conv == my_dol_color * 6).float().sum(dim=(1,2,3)) * 10000000
            total_scores += (conv == my_dol_color * 5).float().sum(dim=(1,2,3)) * 500000
            total_scores += (conv == my_dol_color * 4).float().sum(dim=(1,2,3)) * 10000
            total_scores -= (conv == -my_dol_color * 6).float().sum(dim=(1,2,3)) * 20000000
            total_scores -= (conv == -my_dol_color * 5).float().sum(dim=(1,2,3)) * 1000000
            total_scores -= (conv == -my_dol_color * 4).float().sum(dim=(1,2,3)) * 10000
        
        return total_scores

class minigeneAI:
    def __init__(self):
        self.evaluator = See_borad()
    
    def get_valid_modes(self, board_tensor):
        cpu_board = board_tensor.cpu().squeeze()
        already_set = (cpu_board != 0).nonzero(as_tuple=False)
        cango_board = torch.zeros((BOARD_SIZE, BOARD_SIZE), dtype=torch.bool)
        
        for y, x in already_set:
            y_min, y_max = max(0, y-2), min(BOARD_SIZE, y+3)
            x_min, x_max = max(0, x-2), min(BOARD_SIZE, x+3)
            cango_board[y_min:y_max, x_min:x_max] = True
        
        cango_board[cpu_board != 0] = False
        moves = cango_board.nonzero(as_tuple=False).tolist()
        return moves
    
    def genetic_search(self, board, my_dol_color):
        moves = self.get_valid_modes(board)
        if len(moves) < 2: return 0
        
        population_size = [random.sample(moves, 2) for _ in range(70)]
        best_fit = -float('inf')
        
        for _ in range(5):
            batch = board.unsqueeze(0).repeat(len(population_size), 1, 1).clone()
            for i, m in enumerate(population_size):
                batch[i, m[0][0], m[0][1]] = my_dol_color
                batch[i, m[1][0], m[1][1]] = my_dol_color
            
            scores = self.evaluator.evaluate(batch, my_dol_color)
            best_fit = max(best_fit, torch.max(scores).item())
            
            sort_scores, idx = torch.sort(scores, descending=True)
            survivors = [population_size[i] for i in idx[:15]]
            
            population_size_orgin = len(population_size)
            population_size = survivors[:]
            while len(population_size) < population_size_orgin:
                p1, p2 = random.sample(survivors, 2)
                population_size.append([p1[0], p2[1]])
        
        return best_fit
    
    def minimax_search(self, board_tensor, my_dol_color):
        valid_moves = self.get_valid_modes(board_tensor)
        candidates = []
        for _ in range(100):
            if len(valid_moves) >= 2:
                candidates.append(random.sample(valid_moves, 2))
        
        best_move = None
        best_score = -float('inf')
        
        for move in candidates:
            temp_board = board_tensor.clone()
            temp_board[move[0][0], move[0][1]] = my_dol_color
            temp_board[move[1][0], move[1][1]] = my_dol_color
            
            상대점수 = self.genetic_search(temp_board, -my_dol_color)
            내보드판 = temp_board.unsqueeze(0)
            내보드판현재점수 = self.evaluator.evaluate(내보드판, my_dol_color).item()
            final_score = 내보드판현재점수 - (상대점수 * 1.5)
            
            if final_score > best_score:
                best_score = final_score
                best_move = move
        
        return best_move

class GameWindow:
    def __init__(self, root):
        self.root = root
        self.root.title("Connect6 (6목) - AI vs AI")
        self.root.geometry("800x850")
        
        self.board = torch.zeros((BOARD_SIZE, BOARD_SIZE), dtype=torch.float32).to(DEVICE)
        self.ai = minigeneAI()
        self.selected_moves = []
        self.stones_to_place = 0
        self.current_color = 0
        self.turn_count = 1
        self.game_over = False
        self.start_color = 0
        self.ai_thinking = False
        
        self.initUI()
        self.show_start_dialog()
    
    def initUI(self):
        self.status_label = tk.Label(self.root, text="게임을 시작하세요.", font=("Arial", 12), fg="blue")
        self.status_label.pack(pady=10)
        
        self.canvas = tk.Canvas(self.root, width=700, height=700, bg='#D2B48C', cursor="cross")
        self.canvas.pack(pady=10)
        self.canvas.bind("<Button-1>", self.on_canvas_click)
        
        button_frame = tk.Frame(self.root)
        button_frame.pack(pady=10)
        
        self.undo_btn = tk.Button(button_frame, text="되돌리기", command=self.undo_move, width=15)
        self.undo_btn.grid(row=0, column=0, padx=5)
        
        self.confirm_btn = tk.Button(button_frame, text="확인", command=self.place_stones, width=15)
        self.confirm_btn.grid(row=0, column=1, padx=5)
        
        self.ai_btn = tk.Button(button_frame, text="AI 추천", command=self.get_ai_suggestion, width=15)
        self.ai_btn.grid(row=0, column=2, padx=5)
        
        self.reset_btn = tk.Button(button_frame, text="새 게임", command=self.reset_game, width=15)
        self.reset_btn.grid(row=1, column=0, columnspan=3, padx=5, pady=5)
        
        self.draw_board()
    
    def draw_board(self):
        self.canvas.delete("all")
        
        for i in range(BOARD_SIZE):
            x = 30 + i * CELL_SIZE
            self.canvas.create_line(x, 30, x, 30 + (BOARD_SIZE - 1) * CELL_SIZE, fill="black")
            
            y = 30 + i * CELL_SIZE
            self.canvas.create_line(30, y, 30 + (BOARD_SIZE - 1) * CELL_SIZE, y, fill="black")
        
        board_np = self.board.cpu().numpy()
        for y in range(BOARD_SIZE):
            for x in range(BOARD_SIZE):
                if board_np[y][x] != 0:
                    px = 30 + x * CELL_SIZE
                    py = 30 + y * CELL_SIZE
                    color = "black" if board_np[y][x] == 1 else "white"
                    self.canvas.create_oval(px - 7, py - 7, px + 7, py + 7, fill=color, outline="black", width=2)
        
        for move in self.selected_moves:
            px = 30 + move[1] * CELL_SIZE
            py = 30 + move[0] * CELL_SIZE
            self.canvas.create_rectangle(px - 10, py - 10, px + 10, py + 10, outline="red", width=2)
    
    def on_canvas_click(self, event):
        if self.game_over or self.ai_thinking:
            return
        
        x = (event.x - 30) / CELL_SIZE
        y = (event.y - 30) / CELL_SIZE
        
        if 0 <= x < BOARD_SIZE and 0 <= y < BOARD_SIZE:
            row, col = int(round(y)), int(round(x))
            
            if self.board[row, col] == 0:
                if [row, col] not in self.selected_moves:
                    self.selected_moves.append([row, col])
                    self.update_status()
                    self.draw_board()
    
    def show_start_dialog(self):
        result = messagebox.askyesno("선공 선택", "흑 선공으로 시작하시겠습니까?\n(Yes: 흑 선공, No: 백 선공)")
        self.start_color = 1 if result else -1
        self.start_game()
    
    def start_game(self):
        self.board = torch.zeros((BOARD_SIZE, BOARD_SIZE), dtype=torch.float32).to(DEVICE)
        self.selected_moves = []
        self.game_over = False
        self.turn_count = 1
        self.current_color = self.start_color
        
        center = BOARD_SIZE // 2
        self.board[center, center] = self.start_color
        color_name = "흑(●)" if self.start_color == 1 else "백(○)"
        self.status_label.config(text=f"{color_name}이 선공! 중앙 ({center}, {center})에 착수했습니다.", fg="green")
        
        self.current_color = -self.start_color
        self.turn_count = 2
        self.stones_to_place = 2
        
        self.update_status()
        self.draw_board()
    
    def update_status(self):
        if self.game_over:
            return
        
        color_name = "흑(●)" if self.current_color == 1 else "백(○)"
        self.status_label.config(
            text=f"[{color_name} 차례] 돌 {self.stones_to_place}개를 선택하세요. ({len(self.selected_moves)}/{self.stones_to_place})",
            fg="blue"
        )
    
    def get_ai_suggestion(self):
        if self.game_over or len(self.selected_moves) > 0:
            messagebox.showwarning("경고", "선택된 돌이 있으면 AI 추천을 사용할 수 없습니다.")
            return
        
        self.ai_thinking = True
        self.status_label.config(text="AI가 수를 생각 중입니다...", fg="orange")
        self.root.update()
        
        try:
            color = self.current_color
            moves = self.ai.minimax_search(self.board, color)
            
            if moves:
                self.selected_moves = [list(m) for m in moves]
                self.update_status()
                self.draw_board()
                self.status_label.config(text="AI 추천 수가 표시되었습니다. 확인을 눌러주세요.", fg="purple")
            else:
                messagebox.showwarning("경고", "유효한 수를 찾을 수 없습니다.")
        finally:
            self.ai_thinking = False
    
    def place_stones(self):
        if len(self.selected_moves) != self.stones_to_place:
            messagebox.showwarning("경고", f"{self.stones_to_place}개의 돌을 선택하세요!")
            return
        
        for r, c in self.selected_moves:
            self.board[r, c] = self.current_color
        
        if self.check_win(self.current_color):
            color_name = "흑(●)" if self.current_color == 1 else "백(○)"
            self.status_label.config(text=f"게임 종료! 승리: {color_name}", fg="red")
            self.game_over = True
            self.draw_board()
            return
        
        self.current_color = -self.current_color
        self.turn_count += 1
        self.selected_moves = []
        
        self.update_status()
        self.draw_board()
    
    def undo_move(self):
        if self.selected_moves:
            self.selected_moves.pop()
            self.update_status()
            self.draw_board()
    
    def reset_game(self):
        self.show_start_dialog()
    
    def check_win(self, color):
        b = self.board.cpu().numpy()
        target = 1 if color == 1 else -1
        
        directions = [(0, 1), (1, 0), (1, 1), (1, -1)]
        
        for y in range(BOARD_SIZE):
            for x in range(BOARD_SIZE):
                if b[y][x] != target:
                    continue
                
                for dy, dx in directions:
                    count = 0
                    for k in range(6):
                        ny, nx = y + k * dy, x + k * dx
                        if 0 <= ny < BOARD_SIZE and 0 <= nx < BOARD_SIZE and b[ny][nx] == target:
                            count += 1
                        else:
                            break
                    if count == 6:
                        return True
        return False

if __name__ == '__main__':
    root = tk.Tk()
    app = GameWindow(root)
    root.mainloop()
