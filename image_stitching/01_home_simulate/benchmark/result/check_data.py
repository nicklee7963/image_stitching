import pandas as pd
import os

GLOBAL_CSV = "./result_crawler/crawler_result.csv"

def print_battle_report():
    if not os.path.exists(GLOBAL_CSV):
        print(f"❌ 找不到 {GLOBAL_CSV}，如果你連這個檔案都刪了，那就真的得重跑了！")
        return

    print("讀取歷史資料中...")
    df = pd.read_csv(GLOBAL_CSV)
    
    print("\n" + "="*70)
    print(" 📊 本次測試【各領域 (Theme)】平均表現戰報 (從 CSV 讀取)")
    print("="*70)

    # 針對每個主題單獨印出戰報
    themes_tested = df['Theme'].unique()
    for theme in themes_tested:
        print(f"\n 📌 主題場景：【{theme}】")
        print("-" * 60)
        
        theme_df = df[df['Theme'] == theme]
        avg_theme_df = theme_df.groupby('Method')[['MSE', 'SSIM', 'Frobenius', 'Cosine']].mean().reset_index()
        print(avg_theme_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
        
        # 找出該領域各項指標冠軍
        best_mse = avg_theme_df.loc[avg_theme_df['MSE'].idxmin()]
        best_ssim = avg_theme_df.loc[avg_theme_df['SSIM'].idxmax()]
        best_frob = avg_theme_df.loc[avg_theme_df['Frobenius'].idxmin()]
        best_cos = avg_theme_df.loc[avg_theme_df['Cosine'].idxmax()]
        
        print(f"\n 🏆 【{theme}】領域冠軍：")
        print(f"  🔸 MSE 最低:  \033[92m{best_mse['Method']}\033[0m ({best_mse['MSE']:.4f})")
        print(f"  🔸 SSIM 最高: \033[92m{best_ssim['Method']}\033[0m ({best_ssim['SSIM']:.4f})")
        print(f"  🔸 Frob. 最低:\033[92m{best_frob['Method']}\033[0m ({best_frob['Frobenius']:.4f})")
        print(f"  🔸 Cos. 最高: \033[92m{best_cos['Method']}\033[0m ({best_cos['Cosine']:.4f})")
        print("*" * 70)

    print("\n\n" + "="*70)
    print(" 🌍 本次測試【綜合總平均】跨領域決選戰報")
    print("="*70)
    
    # 依照 Method 計算全部資料的平均
    avg_df = df.groupby('Method')[['MSE', 'SSIM', 'Frobenius', 'Cosine']].mean().reset_index()
    print(avg_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print("-" * 70)
    
    # 找出總冠軍
    best_overall_mse = avg_df.loc[avg_df['MSE'].idxmin()]
    best_overall_ssim = avg_df.loc[avg_df['SSIM'].idxmax()]
    best_overall_frob = avg_df.loc[avg_df['Frobenius'].idxmin()]
    best_overall_cos = avg_df.loc[avg_df['Cosine'].idxmax()]
    
    print(" 👑 各項指標【全領域綜合總冠軍】：")
    print(f"  🔸 影像復原度 (MSE 越低越好):  \033[96m{best_overall_mse['Method']}\033[0m")
    print(f"  🔸 結構相似度 (SSIM 越高越好): \033[96m{best_overall_ssim['Method']}\033[0m")
    print(f"  🔸 幾何精準度 (Frob. 越低越好): \033[96m{best_overall_frob['Method']}\033[0m")
    print(f"  🔸 矩陣相似度 (Cos. 越高越好):  \033[96m{best_overall_cos['Method']}\033[0m")
    print("="*70)

if __name__ == "__main__":
    print_battle_report()
