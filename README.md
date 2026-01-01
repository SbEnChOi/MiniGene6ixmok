<img width="1238" height="1457" alt="image" src="https://github.com/user-attachments/assets/973ccb22-0920-4e9f-8267-e3e125a862ab" />

현재 하이퍼파라미터 세팅 
-MiniMax level(depth) : 1
-Gentic algorithm 해집단 수 : 100
-final Score 가중치 : 내보드판현재점수 - (상대점수 * 0.5) # 이1.5 를 조절해서 공격적으로 or 방어적으로로 할 수있음
- 평가 함수
 total_scores += (conv == my_dol_color * 6).float().sum(dim=(1,2,3)) * 10000000
            total_scores += (conv == my_dol_color * 5).float().sum(dim=(1,2,3)) * 500000
            total_scores += (conv == my_dol_color * 4).float().sum(dim=(1,2,3)) * 10000
            # 방어 점수
            total_scores -= (conv == -my_dol_color * 6).float().sum(dim=(1,2,3)) * 20000000
            total_scores -= (conv == -my_dol_color * 5).float().sum(dim=(1,2,3)) * 1000000
            total_scores -= (conv == -my_dol_color * 4).float().sum(dim=(1,2,3)) * 10000
        
