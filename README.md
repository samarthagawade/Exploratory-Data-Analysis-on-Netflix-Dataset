# 📊 Netflix Data Analysis Project

## 🏷️ **Project Title**
**Exploratory Data Analysis on Netflix Dataset**

---

## 🎯 **Objective**
The objective of this project is to analyze the Netflix dataset to uncover insights about the platform’s content distribution, genres, release trends, and audience ratings.  
Through this analysis, we aim to understand:  
- How Netflix’s content library has evolved over the years  
- Which genres and countries dominate the platform  
- What type of content (Movies or TV Shows) is more prevalent  

---

## 📁 **Dataset Description**
- **Dataset Name:** Netflix Titles Dataset  
- **File Format:** CSV  
- **Columns Include:**  
  `show_id`, `type`, `title`, `director`, `cast`, `country`, `date_added`, `release_year`, `rating`, `duration`, `listed_in`, `description`  
- **Records:** ~8,800 titles  
- **Time Range:** 1925 – 2021  

### 🔍 Key Features:
- Contains both **Movies** and **TV Shows**  
- Includes metadata such as **director, cast, rating, genre, and duration**  
- Missing values handled using imputation or “Unknown” placeholders  

---

## ⚙️ **Methodology**
1. **Data Cleaning**
   - Loaded data using `pandas`
   - Handled null and inconsistent values
   - Converted `date_added` into `datetime` format
   - Extracted `year_added` for time-based analysis  

2. **Exploratory Data Analysis (EDA)**
   - Analyzed distributions of content by type, genre, and country  
   - Identified top contributing countries and most popular genres  
   - Examined release year trends and audience ratings  

3. **Visualization**
   - Used `matplotlib` and `seaborn` for graphical representation  
   - Created bar plots, count plots, and trend lines for better insights  

---

## 📈 **Key Findings**
- **Movies (≈ 70%)** dominate over **TV Shows (≈ 30%)**.  
- **Drama**, **Comedy**, and **Documentary** are the most common genres.  
- The **United States** and **India** produce the majority of Netflix titles.  
- Most titles were released between **2015–2021**, indicating rapid expansion.  
- Common content ratings are **TV-MA**, **TV-14**, and **TV-PG**.  

---

## 🧩 **Tools & Technologies Used**
- **Programming Language:** Python  
- **Libraries:**  
  - `pandas` – Data loading and preprocessing  
  - `numpy` – Numerical computations  
  - `matplotlib`, `seaborn` – Data visualization  
- **Platform:** Google Colab  

---

## 🧠 **Conclusion**
The analysis provides valuable insights into Netflix’s global content strategy.  
Movies dominate the catalog, and the U.S. and India are top content contributors.  
Netflix’s strongest growth occurred after 2015, reflecting its expansion into global markets.  
Future work could include building a **content-based recommendation system** or integrating **viewer ratings** for popularity analysis.

---
