# R for Big Data Wrangling: Python/PySpark to R Transition
## 3-Hour Training Session

**Target Audience:** Data scientists proficient in Python, PySpark, and big data wrangling  
**Goal:** Master R equivalents using Tidyverse, dplyr, dbplyr, data.table, and IBM DB2 integration  
**Duration:** 3 hours (180 minutes)

---

## Table of Contents
1. [Setup & Environment (15 min)](#session-1-setup--environment-15-min)
2. [Core Tidyverse & dplyr Fundamentals (45 min)](#session-2-core-tidyverse--dplyr-fundamentals-45-min)
3. [High-Performance data.table (30 min)](#session-3-high-performance-datatable-30-min)
4. [Database Connectivity with dbplyr & IBM DB2 (40 min)](#session-4-database-connectivity-with-dbplyr--ibm-db2-40-min)
5. [Advanced Patterns & Best Practices (30 min)](#session-5-advanced-patterns--best-practices-30-min)
6. [Python/PySpark to R Cheatsheet](#pythonpyspark-to-r-cheatsheet)
7. [Resources & Next Steps (20 min)](#session-6-resources--next-steps-20-min)

---

## Session 1: Setup & Environment (15 min)

### Required Packages Installation

```r
# Core data manipulation packages
install.packages(c(
  "tidyverse",      # Meta-package: dplyr, tidyr, ggplot2, etc.
  "data.table",     # High-performance data manipulation
  "dbplyr",         # Database backend for dplyr
  "DBI",            # Database interface
  "RJDBC",          # JDBC connectivity for DB2
  "dtplyr",         # data.table backend for dplyr syntax
  "microbenchmark", # Performance testing
  "arrow",          # For parquet files (like PySpark)
  "bench"           # Benchmarking tools
))
```

### Loading Libraries

```r
library(tidyverse)   # Loads dplyr, tidyr, ggplot2, readr, etc.
library(data.table)
library(dbplyr)
library(DBI)
```

### Key Philosophy Differences

| Concept | Python/Pandas | R Tidyverse | R data.table |
|---------|---------------|-------------|--------------|
| **Syntax Style** | Object-oriented (df.method()) | Functional/Piped (df %>% verb()) | Modified in-place [DT, action] |
| **Immutability** | Generally immutable | Immutable (creates copies) | Can modify by reference |
| **Chaining** | Method chaining | Pipe operator %>% or \|> | Chained brackets [...][...] |
| **Performance** | Moderate | Good | Excellent (C-optimized) |

---

## Session 2: Core Tidyverse & dplyr Fundamentals (45 min)

### The Pipe Operator: Your New Best Friend

The pipe `%>%` (or native `|>` in R >= 4.1) replaces method chaining:

```r
# Python/Pandas style (conceptual)
# df.filter(...).groupby(...).agg(...)

# R Tidyverse style
df %>%
  filter(...) %>%
  group_by(...) %>%
  summarize(...)
```

### Core dplyr Verbs (The "Grammar" of Data Manipulation)

#### 1. **filter()** - Row Selection (like pandas query/boolean indexing)

```r
# Python: df[df['age'] > 30]
# PySpark: df.filter(col('age') > 30)

# R dplyr:
employees %>%
  filter(age > 30)

# Multiple conditions
employees %>%
  filter(age > 30, department == "Sales")

# OR conditions
employees %>%
  filter(age > 30 | salary > 100000)

# Using %in% (like Python's .isin())
employees %>%
  filter(department %in% c("Sales", "Marketing", "IT"))
```

#### 2. **select()** - Column Selection

```r
# Python: df[['name', 'age', 'salary']]
# PySpark: df.select('name', 'age', 'salary')

# R dplyr:
employees %>%
  select(name, age, salary)

# Helper functions
employees %>%
  select(starts_with("emp_"))       # Like df.filter(regex='^emp_')

employees %>%
  select(ends_with("_date"))

employees %>%
  select(contains("addr"))          # Contains substring

employees %>%
  select(matches("^[A-Z]"))        # Regex pattern

employees %>%
  select(where(is.numeric))         # Select by type

# Exclude columns
employees %>%
  select(-c(ssn, password))         # Drop columns
```

#### 3. **mutate()** - Create/Modify Columns

```r
# Python: df['total_comp'] = df['salary'] + df['bonus']
# PySpark: df.withColumn('total_comp', col('salary') + col('bonus'))

# R dplyr:
employees %>%
  mutate(total_comp = salary + bonus)

# Multiple columns at once
employees %>%
  mutate(
    total_comp = salary + bonus,
    monthly_salary = salary / 12,
    salary_k = salary / 1000,
    is_senior = age > 40
  )

# Conditional mutations (like np.where)
employees %>%
  mutate(
    experience_level = case_when(
      years_exp < 3 ~ "Junior",
      years_exp < 7 ~ "Mid",
      years_exp < 12 ~ "Senior",
      TRUE ~ "Principal"  # default case
    )
  )

# Using across() for multiple columns (vectorized operations)
employees %>%
  mutate(across(where(is.numeric), ~ . / 1000, .names = "{.col}_k"))
```

#### 4. **arrange()** - Sorting

```r
# Python: df.sort_values('salary', ascending=False)
# PySpark: df.orderBy(col('salary').desc())

# R dplyr:
employees %>%
  arrange(salary)                   # Ascending

employees %>%
  arrange(desc(salary))             # Descending

# Multiple columns
employees %>%
  arrange(department, desc(salary))
```

#### 5. **group_by() + summarize()** - Aggregation

```r
# Python: df.groupby('department')['salary'].mean()
# PySpark: df.groupBy('department').agg(avg('salary'))

# R dplyr:
employees %>%
  group_by(department) %>%
  summarize(avg_salary = mean(salary))

# Multiple aggregations
employees %>%
  group_by(department) %>%
  summarize(
    count = n(),                    # Row count
    avg_salary = mean(salary),
    median_salary = median(salary),
    total_payroll = sum(salary),
    std_dev = sd(salary),
    min_sal = min(salary),
    max_sal = max(salary)
  )

# Multiple grouping columns
employees %>%
  group_by(department, location) %>%
  summarize(
    count = n(),
    avg_salary = mean(salary, na.rm = TRUE)  # Handle NAs
  )

# Using across() for multiple columns
employees %>%
  group_by(department) %>%
  summarize(across(
    c(salary, bonus),
    list(mean = mean, sd = sd),
    .names = "{.col}_{.fn}"
  ))
```

#### 6. **Window Functions** (like PySpark window operations)

```r
# Python/PySpark window function
# window = Window.partitionBy('department').orderBy('salary')
# df.withColumn('rank', rank().over(window))

# R dplyr:
employees %>%
  group_by(department) %>%
  mutate(
    rank = row_number(desc(salary)),
    dense_rank = dense_rank(desc(salary)),
    salary_percentile = percent_rank(salary),
    salary_ntile = ntile(salary, 4),  # Quartiles
    dept_avg = mean(salary),
    diff_from_avg = salary - mean(salary),
    cumulative_sum = cumsum(salary),
    lead_salary = lead(salary),
    lag_salary = lag(salary)
  ) %>%
  ungroup()  # Important: ungroup after window operations!
```

### Practical Exercise 1: Data Transformation Pipeline (15 min)

```r
# Sample dataset
set.seed(42)
sales_data <- tibble(
  transaction_id = 1:1000,
  customer_id = sample(1:200, 1000, replace = TRUE),
  product_category = sample(c("Electronics", "Clothing", "Food", "Books"), 1000, replace = TRUE),
  amount = rnorm(1000, mean = 100, sd = 30),
  date = seq.Date(from = as.Date("2024-01-01"), by = "day", length.out = 1000)
)

# Task: Create a summary report
customer_summary <- sales_data %>%
  filter(amount > 0) %>%
  mutate(
    month = floor_date(date, "month"),
    amount_category = case_when(
      amount < 50 ~ "Low",
      amount < 100 ~ "Medium",
      amount < 150 ~ "High",
      TRUE ~ "Premium"
    )
  ) %>%
  group_by(customer_id, month, product_category) %>%
  summarize(
    total_spent = sum(amount),
    transaction_count = n(),
    avg_transaction = mean(amount),
    .groups = "drop"  # Automatically ungroup
  ) %>%
  arrange(desc(total_spent))

# View results
head(customer_summary, 10)
```

---

## Session 3: High-Performance data.table (30 min)

### Why data.table?

- **Speed**: 10-100x faster than dplyr for large datasets (>1M rows)
- **Memory Efficient**: Modifies by reference (no copies)
- **Syntax**: More concise but steeper learning curve
- **Best For**: Large datasets, production ETL, memory-constrained environments

### data.table Syntax: DT[i, j, by]

```
DT[i,  j,  by]
   |   |   |
   |   |   +----- Group by
   |   +--------- Select/Compute columns  
   +------------- Filter rows
```

### Core Operations Comparison

#### Creating a data.table

```r
library(data.table)

# From data.frame
dt <- as.data.table(df)

# Direct creation
dt <- data.table(
  id = 1:5,
  name = c("Alice", "Bob", "Charlie", "David", "Eve"),
  age = c(25, 30, 35, 40, 45)
)

# Read from file (very fast!)
dt <- fread("large_file.csv")  # Much faster than read.csv()
```

#### Filtering (i position)

```r
# Python: df[df['age'] > 30]
# dplyr: filter(df, age > 30)

# data.table:
dt[age > 30]

# Multiple conditions
dt[age > 30 & department == "Sales"]

# Using %in%
dt[department %in% c("Sales", "IT")]

# Using %between%
dt[age %between% c(25, 35)]
```

#### Selecting & Computing Columns (j position)

```r
# Select columns
# Python: df[['name', 'age']]
# dplyr: select(df, name, age)

# data.table:
dt[, .(name, age)]                    # .() is alias for list()
dt[, list(name, age)]                 # Same thing

# Compute new columns
# Python: df['total'] = df['salary'] + df['bonus']
# dplyr: mutate(df, total = salary + bonus)

# data.table (creates new column by reference!):
dt[, total_comp := salary + bonus]

# Multiple columns
dt[, `:=`(
  total_comp = salary + bonus,
  monthly_sal = salary / 12
)]

# Conditional column
dt[, experience_level := fcase(
  years_exp < 3, "Junior",
  years_exp < 7, "Mid",
  years_exp < 12, "Senior",
  default = "Principal"
)]
```

#### Aggregation (by position)

```r
# Python: df.groupby('department')['salary'].mean()
# dplyr: df %>% group_by(department) %>% summarize(avg = mean(salary))

# data.table:
dt[, .(avg_salary = mean(salary)), by = department]

# Multiple aggregations
dt[, .(
  count = .N,                      # .N is special symbol for row count
  avg_salary = mean(salary),
  total_payroll = sum(salary),
  median_sal = median(salary)
), by = department]

# Multiple grouping columns
dt[, .(avg_salary = mean(salary)), by = .(department, location)]

# Using .SD (Subset of Data)
dt[, lapply(.SD, mean), by = department, .SDcols = c("salary", "bonus")]
```

#### Sorting (order in i position)

```r
# Python: df.sort_values('salary', ascending=False)
# dplyr: arrange(df, desc(salary))

# data.table:
dt[order(-salary)]                   # - for descending

# Multiple columns
dt[order(department, -salary)]
```

#### Chaining Operations

```r
# dplyr uses %>%
# data.table chains with ][][]

dt[age > 30][
  , .(avg_salary = mean(salary)), by = department
][
  order(-avg_salary)
][
  1:5  # Top 5
]
```

### Keys and Indices (Super-Fast Lookups!)

```r
# Set key for ultra-fast filtering (like database index)
setkey(dt, customer_id, date)

# Now filtering on key is VERY fast (binary search)
dt[.(123, as.Date("2024-01-01"))]  # Lightning fast lookup

# Set secondary indices
setindex(dt, product_category)
setindex(dt, department)

# View keys and indices
key(dt)
indices(dt)
```

### Joins (Very Fast!)

```r
# Python: pd.merge(left, right, on='id', how='left')
# dplyr: left_join(left, right, by = "id")

# data.table:
setkey(dt1, id)
setkey(dt2, id)

dt1[dt2]                             # Right join
dt2[dt1]                             # Left join
merge(dt1, dt2, by = "id")          # Inner join
merge(dt1, dt2, by = "id", all.x = TRUE)   # Left join
merge(dt1, dt2, by = "id", all = TRUE)     # Full outer join
```

### Practical Exercise 2: data.table Performance (10 min)

```r
library(data.table)
library(microbenchmark)

# Create large dataset
n <- 1e6
large_dt <- data.table(
  id = sample(1:10000, n, replace = TRUE),
  category = sample(LETTERS[1:5], n, replace = TRUE),
  value1 = rnorm(n),
  value2 = rnorm(n)
)

# Benchmark: Aggregation
microbenchmark(
  data.table = large_dt[, .(
    mean_v1 = mean(value1),
    mean_v2 = mean(value2),
    count = .N
  ), by = .(id, category)],
  
  dplyr = large_dt %>%
    group_by(id, category) %>%
    summarize(
      mean_v1 = mean(value1),
      mean_v2 = mean(value2),
      count = n(),
      .groups = "drop"
    ),
  
  times = 10
)

# Set key for fast filtering
setkey(large_dt, id, category)

# Benchmark: Filtering with key
microbenchmark(
  with_key = large_dt[.(1234, "A")],
  without_key = large_dt[id == 1234 & category == "A"],
  times = 100
)
```

---

## Session 4: Database Connectivity with dbplyr & IBM DB2 (40 min)

### Understanding dbplyr

**dbplyr** translates dplyr code into SQL queries. You write R, it executes SQL on the database!

```r
# Benefits:
# 1. Use dplyr syntax on database tables
# 2. Computation happens on database server (not R)
# 3. Only brings results to R when needed (lazy evaluation)
# 4. Automatic SQL optimization
```

### Setting Up IBM DB2 Connection

#### Option 1: Using RJDBC (Recommended for DB2)

```r
library(DBI)
library(RJDBC)

# Download IBM DB2 JDBC driver first
# Place db2jcc4.jar in a known location

# Initialize JDBC driver
db2_driver <- JDBC(
  driverClass = "com.ibm.db2.jcc.DB2Driver",
  classPath = "/path/to/db2jcc4.jar"
)

# Create connection
con <- dbConnect(
  db2_driver,
  "jdbc:db2://hostname:50000/database_name",
  user = "your_username",
  password = "your_password"
)

# Test connection
dbGetQuery(con, "SELECT CURRENT TIMESTAMP FROM SYSIBM.SYSDUMMY1")
```

#### Option 2: Using odbc (Alternative)

```r
library(DBI)
library(odbc)

# Requires IBM DB2 ODBC driver installed
con <- dbConnect(
  odbc::odbc(),
  Driver = "IBM DB2 ODBC DRIVER",
  Database = "database_name",
  Hostname = "hostname",
  Port = 50000,
  Protocol = "TCPIP",
  UID = "your_username",
  PWD = "your_password"
)
```

### Working with Database Tables using dbplyr

```r
library(dplyr)
library(dbplyr)

# Reference a table (doesn't load data!)
employees_tbl <- tbl(con, "EMPLOYEES")
sales_tbl <- tbl(con, "SALES_TRANSACTIONS")

# You can also use schema
employees_tbl <- tbl(con, in_schema("HR_SCHEMA", "EMPLOYEES"))

# View table structure
glimpse(employees_tbl)
```

### Lazy Evaluation - The Magic of dbplyr

```r
# This doesn't execute on DB yet - builds a query plan
query <- employees_tbl %>%
  filter(department == "Sales") %>%
  select(employee_id, name, salary) %>%
  arrange(desc(salary))

# See the SQL that will be generated
show_query(query)

# Execution only happens when you:
# 1. collect() - Bring all data to R
result <- query %>% collect()

# 2. compute() - Create temp table on database
temp_result <- query %>% compute()

# 3. head() - Bring first n rows
preview <- query %>% head(10)

# 4. Print/view the object
print(query)  # Fetches preview
```

### Key dbplyr Operations

```r
# Filtering
sales_tbl %>%
  filter(
    transaction_date >= "2024-01-01",
    transaction_date < "2024-04-01",
    amount > 100
  )

# Aggregation (executes SUM on database!)
sales_tbl %>%
  group_by(customer_id, product_category) %>%
  summarize(
    total_sales = sum(amount, na.rm = TRUE),
    transaction_count = n(),
    avg_sale = mean(amount, na.rm = TRUE)
  ) %>%
  arrange(desc(total_sales))

# Window functions (translates to DB window functions!)
sales_tbl %>%
  group_by(customer_id) %>%
  mutate(
    running_total = cumsum(amount),
    transaction_rank = row_number(desc(amount)),
    pct_of_customer_total = amount / sum(amount)
  )

# Joins
employees_tbl %>%
  left_join(
    sales_tbl,
    by = c("employee_id" = "salesperson_id")
  ) %>%
  group_by(employee_id, name) %>%
  summarize(total_sales = sum(amount, na.rm = TRUE))
```

### Advanced DB2-Specific Operations

#### Using SQL Directly

```r
# When dbplyr syntax doesn't cut it
custom_query <- dbGetQuery(con, "
  SELECT 
    department,
    COUNT(*) as employee_count,
    AVG(salary) as avg_salary,
    PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY salary) as median_salary
  FROM HR_SCHEMA.EMPLOYEES
  GROUP BY department
  HAVING AVG(salary) > 75000
  ORDER BY avg_salary DESC
")
```

#### Hybrid Approach: Start with SQL, Continue with dplyr

```r
# Complex SQL for initial extraction
initial_query <- "
  WITH ranked_sales AS (
    SELECT 
      s.*,
      ROW_NUMBER() OVER (PARTITION BY customer_id ORDER BY amount DESC) as rn
    FROM SALES_SCHEMA.TRANSACTIONS s
    WHERE transaction_date >= CURRENT_DATE - 90 DAYS
  )
  SELECT * FROM ranked_sales WHERE rn <= 5
"

# Load initial results and continue processing in R
top_sales <- tbl(con, sql(initial_query)) %>%
  mutate(amount_usd = amount * exchange_rate) %>%
  group_by(customer_id) %>%
  summarize(
    top5_total = sum(amount_usd),
    categories = n_distinct(product_category)
  ) %>%
  collect()
```

### Writing Data Back to Database

```r
# Write R data.frame to database table
local_data <- data.frame(
  id = 1:100,
  value = rnorm(100)
)

# Create new table
dbWriteTable(con, "MY_NEW_TABLE", local_data, overwrite = TRUE)

# Append to existing table
dbWriteTable(con, "MY_NEW_TABLE", local_data, append = TRUE)

# Using copy_to with dbplyr (creates temp table)
temp_tbl <- copy_to(con, local_data, "TEMP_DATA", temporary = TRUE)
```

### Performance Best Practices

```r
# 1. Filter early and often (pushdown predicates)
# GOOD:
result <- large_table %>%
  filter(date >= "2024-01-01") %>%     # Filter first
  select(id, amount) %>%                # Select needed columns
  group_by(id) %>%
  summarize(total = sum(amount)) %>%
  collect()

# BAD:
result <- large_table %>%
  collect() %>%                         # Brings ALL data to R first!
  filter(date >= "2024-01-01") %>%
  group_by(id) %>%
  summarize(total = sum(amount))

# 2. Use compute() for reused intermediate results
expensive_query <- large_table %>%
  filter(complex_condition) %>%
  compute(name = "temp_filtered", temporary = TRUE)

# Now reuse without re-running
result1 <- expensive_query %>% summarize(count = n())
result2 <- expensive_query %>% summarize(avg_val = mean(amount))

# 3. Check query plans
my_query %>% show_query()            # See SQL
my_query %>% explain()               # Database query plan

# 4. Use indices on database side for better performance
dbGetQuery(con, "CREATE INDEX idx_date ON SALES(transaction_date)")
```

### Practical Exercise 3: End-to-End DB2 Workflow (15 min)

```r
# Scenario: Analyze customer purchase patterns from DB2

# 1. Connect to DB2
con <- dbConnect(
  RJDBC::JDBC(
    driverClass = "com.ibm.db2.jcc.DB2Driver",
    classPath = "/path/to/db2jcc4.jar"
  ),
  "jdbc:db2://hostname:50000/SALES_DB",
  user = "analyst",
  password = Sys.getenv("DB2_PASSWORD")  # Secure password handling
)

# 2. Reference tables
customers_tbl <- tbl(con, in_schema("RETAIL", "CUSTOMERS"))
transactions_tbl <- tbl(con, in_schema("RETAIL", "TRANSACTIONS"))
products_tbl <- tbl(con, in_schema("RETAIL", "PRODUCTS"))

# 3. Build analysis query (all executed on DB2!)
customer_analysis <- transactions_tbl %>%
  filter(
    transaction_date >= "2024-01-01",
    transaction_date < "2024-04-01"
  ) %>%
  left_join(products_tbl, by = "product_id") %>%
  left_join(customers_tbl, by = "customer_id") %>%
  group_by(customer_id, customer_segment, region) %>%
  summarize(
    total_spent = sum(amount, na.rm = TRUE),
    transaction_count = n(),
    unique_products = n_distinct(product_id),
    avg_transaction = mean(amount, na.rm = TRUE),
    .groups = "drop"
  ) %>%
  filter(total_spent > 1000) %>%
  arrange(desc(total_spent))

# 4. Check the generated SQL
show_query(customer_analysis)

# 5. Execute and bring to R for visualization
results <- customer_analysis %>% collect()

# 6. Further processing in R (now that data is local)
library(ggplot2)

results %>%
  mutate(
    value_segment = case_when(
      total_spent >= 10000 ~ "VIP",
      total_spent >= 5000 ~ "High Value",
      total_spent >= 2000 ~ "Medium Value",
      TRUE ~ "Standard"
    )
  ) %>%
  ggplot(aes(x = transaction_count, y = total_spent, color = value_segment)) +
  geom_point(alpha = 0.6) +
  scale_y_log10() +
  theme_minimal() +
  labs(
    title = "Customer Value Analysis",
    x = "Transaction Count",
    y = "Total Spent (log scale)"
  )

# 7. Clean up
dbDisconnect(con)
```

---

## Session 5: Advanced Patterns & Best Practices (30 min)

### When to Use Which Tool?

```r
# Decision Tree:
#
# Data size < 10K rows?
# └─> Use dplyr (readability, ease of use)
#
# Data size 10K - 1M rows?
# └─> Use dplyr (good balance of speed and readability)
#     └─> If performance issues: try dtplyr (dplyr syntax, data.table backend)
#
# Data size > 1M rows in memory?
# └─> Use data.table (raw speed and memory efficiency)
#
# Data size > RAM or in database?
# └─> Use dbplyr (let database do the work)
#
# Need to share code with non-R users?
# └─> Use dplyr (more readable, easier to understand)
#
# Production ETL pipeline?
# └─> Use data.table (speed, reliability, memory efficiency)
```

### Combining Tools: dtplyr (Best of Both Worlds!)

```r
library(dtplyr)

# Use dplyr syntax with data.table performance
dt <- as.data.table(large_df)

result <- dt %>%
  lazy_dt() %>%                        # Convert to lazy data.table
  filter(amount > 100) %>%
  group_by(category) %>%
  summarize(
    total = sum(amount),
    count = n()
  ) %>%
  arrange(desc(total)) %>%
  as_tibble()                          # Convert back to tibble

# See the generated data.table code
dt %>%
  lazy_dt() %>%
  filter(amount > 100) %>%
  show_query()
```

### Efficient Data Import/Export

```r
# CSV Files
# =========

# Slow (base R)
df <- read.csv("large_file.csv")

# Fast (readr - part of tidyverse)
df <- read_csv("large_file.csv")

# Fastest (data.table)
dt <- fread("large_file.csv")

# Write
fwrite(dt, "output.csv")             # data.table - very fast
write_csv(df, "output.csv")          # readr


# Parquet Files (like PySpark!)
# =============================

library(arrow)

# Read parquet (like spark.read.parquet())
df <- read_parquet("file.parquet")

# Write parquet
write_parquet(df, "output.parquet")

# Work with partitioned parquet datasets
dataset <- open_dataset("partitioned_data/")

dataset %>%
  filter(year == 2024, month %in% c(1, 2, 3)) %>%
  select(id, amount, date) %>%
  group_by(category) %>%
  summarize(total = sum(amount)) %>%
  collect()


# Binary R formats (fastest for R-to-R)
# ======================================

# RDS (single object)
saveRDS(large_df, "data.rds")
large_df <- readRDS("data.rds")

# RData (multiple objects)
save(df1, df2, df3, file = "data.RData")
load("data.RData")

# FST format (extremely fast!)
library(fst)
write_fst(df, "data.fst")
df <- read_fst("data.fst")
```

### Memory Management

```r
# Check object sizes
object.size(large_dt)
pryr::object_size(large_dt)          # More accurate

# List large objects
sort(sapply(ls(), function(x) object.size(get(x))), decreasing = TRUE)

# Remove objects and free memory
rm(large_dt)
gc()                                  # Garbage collection

# Efficient processing of large files in chunks
library(readr)

# Process CSV in chunks
chunked_result <- read_csv_chunked(
  "huge_file.csv",
  callback = DataFrameCallback$new(function(chunk, pos) {
    # Process each chunk
    chunk %>%
      filter(amount > 100) %>%
      group_by(category) %>%
      summarize(total = sum(amount))
  }),
  chunk_size = 10000
)
```

### Functional Programming Patterns

```r
# purrr for iteration (part of tidyverse)
library(purrr)

# Apply function to each element (like Python map)
# Python: list(map(sqrt, [1, 4, 9, 16]))
# R:
map_dbl(c(1, 4, 9, 16), sqrt)

# Read multiple files
file_paths <- list.files("data/", pattern = "*.csv", full.names = TRUE)

all_data <- map_dfr(file_paths, read_csv)  # Combine by rows

# Nested data manipulation
nested_data <- transactions %>%
  group_by(customer_id) %>%
  nest() %>%                           # Creates list-column of data.frames
  mutate(
    model = map(data, ~lm(amount ~ date, data = .)),
    predictions = map2(model, data, predict),
    metrics = map(model, broom::glance)
  ) %>%
  unnest(metrics)

# Safely handle errors
safe_log <- safely(log)
results <- map(c(10, -5, 0, 100), safe_log)

# Extract results and errors
map(results, "result")
map(results, "error")
```

### Production-Ready Code Patterns

```r
# 1. Use projects and relative paths
library(here)
data_path <- here("data", "raw", "input.csv")

# 2. Configuration management
config <- list(
  db_host = Sys.getenv("DB_HOST"),
  db_user = Sys.getenv("DB_USER"),
  db_password = Sys.getenv("DB_PASSWORD"),
  chunk_size = 10000,
  output_path = here("data", "processed")
)

# 3. Logging
library(logger)

log_info("Starting data processing pipeline")
log_debug("Reading data from {data_path}")
log_warn("Found {n_missing} missing values")
log_error("Database connection failed")

# 4. Error handling
tryCatch({
  result <- risky_operation()
}, error = function(e) {
  log_error("Operation failed: {e$message}")
  # Fallback or cleanup
}, finally = {
  # Cleanup code (always runs)
  dbDisconnect(con)
})

# 5. Function documentation
#' Calculate customer lifetime value
#'
#' @param transactions Data frame with customer transactions
#' @param customer_id Customer identifier column name
#' @param amount Amount column name
#' @return Data frame with CLV metrics
#' @export
calculate_clv <- function(transactions, customer_id, amount) {
  transactions %>%
    group_by({{ customer_id }}) %>%
    summarize(
      total_value = sum({{ amount }}),
      transaction_count = n(),
      avg_value = mean({{ amount }})
    )
}
```

---

## Python/PySpark to R Cheatsheet

### Data Structures

| Python/PySpark | R dplyr | R data.table |
|----------------|---------|--------------|
| `pd.DataFrame()` | `tibble()` or `data.frame()` | `data.table()` |
| `spark.createDataFrame()` | `as_tibble()` | `as.data.table()` |
| `df.columns` | `names(df)` or `colnames(df)` | `names(dt)` or `colnames(dt)` |
| `df.shape` | `dim(df)` | `dim(dt)` |
| `df.dtypes` | `str(df)` or `glimpse(df)` | `str(dt)` |
| `df.head()` | `head(df)` | `head(dt)` |
| `df.describe()` | `summary(df)` | `summary(dt)` |

### Reading Data

| Python/PySpark | R dplyr/readr | R data.table |
|----------------|---------------|--------------|
| `pd.read_csv('file.csv')` | `read_csv('file.csv')` | `fread('file.csv')` |
| `spark.read.parquet('file.parquet')` | `read_parquet('file.parquet')` | `read_parquet('file.parquet')` |
| `pd.read_json('file.json')` | `read_json('file.json')` | `fread('file.json')` |
| `spark.read.jdbc(url, table)` | `tbl(con, 'table')` | `dbReadTable(con, 'table')` |

### Filtering (Row Selection)

| Python/PySpark | R dplyr | R data.table |
|----------------|---------|--------------|
| `df[df['age'] > 30]` | `filter(df, age > 30)` | `dt[age > 30]` |
| `df.query('age > 30')` | `filter(df, age > 30)` | `dt[age > 30]` |
| `df.filter(col('age') > 30)` | `filter(df, age > 30)` | `dt[age > 30]` |
| `df[df['col'].isin([1,2,3])]` | `filter(df, col %in% c(1,2,3))` | `dt[col %in% c(1,2,3)]` |
| `df[(df['a'] > 5) & (df['b'] < 10)]` | `filter(df, a > 5, b < 10)` | `dt[a > 5 & b < 10]` |
| `df[df['age'].between(20, 30)]` | `filter(df, between(age, 20, 30))` | `dt[age %between% c(20, 30)]` |

### Selecting (Column Selection)

| Python/PySpark | R dplyr | R data.table |
|----------------|---------|--------------|
| `df[['col1', 'col2']]` | `select(df, col1, col2)` | `dt[, .(col1, col2)]` |
| `df.select('col1', 'col2')` | `select(df, col1, col2)` | `dt[, .(col1, col2)]` |
| `df.drop('col1')` | `select(df, -col1)` | `dt[, !"col1"]` |
| `df.filter(regex='^col')` | `select(df, starts_with('col'))` | `dt[, .SD, .SDcols = patterns('^col')]` |
| `df.select_dtypes(include=['number'])` | `select(df, where(is.numeric))` | `dt[, .SD, .SDcols = is.numeric]` |

### Creating/Modifying Columns

| Python/PySpark | R dplyr | R data.table |
|----------------|---------|--------------|
| `df['new_col'] = df['a'] + df['b']` | `mutate(df, new_col = a + b)` | `dt[, new_col := a + b]` |
| `df.withColumn('new_col', col('a') + col('b'))` | `mutate(df, new_col = a + b)` | `dt[, new_col := a + b]` |
| `df.assign(new_col = df['a'] + df['b'])` | `mutate(df, new_col = a + b)` | `dt[, new_col := a + b]` |
| `np.where(df['a'] > 5, 'high', 'low')` | `mutate(df, cat = ifelse(a > 5, 'high', 'low'))` | `dt[, cat := ifelse(a > 5, 'high', 'low')]` |
| `df.withColumn('cat', when(col('a') > 5, 'high').otherwise('low'))` | `mutate(df, cat = case_when(a > 5 ~ 'high', TRUE ~ 'low'))` | `dt[, cat := fcase(a > 5, 'high', default='low')]` |

### Sorting

| Python/PySpark | R dplyr | R data.table |
|----------------|---------|--------------|
| `df.sort_values('col')` | `arrange(df, col)` | `dt[order(col)]` |
| `df.sort_values('col', ascending=False)` | `arrange(df, desc(col))` | `dt[order(-col)]` |
| `df.orderBy('col')` | `arrange(df, col)` | `dt[order(col)]` |
| `df.orderBy(col('col').desc())` | `arrange(df, desc(col))` | `dt[order(-col)]` |
| `df.sort_values(['a', 'b'], ascending=[True, False])` | `arrange(df, a, desc(b))` | `dt[order(a, -b)]` |

### Aggregation

| Python/PySpark | R dplyr | R data.table |
|----------------|---------|--------------|
| `df.groupby('col')['val'].mean()` | `df %>% group_by(col) %>% summarize(mean = mean(val))` | `dt[, .(mean = mean(val)), by = col]` |
| `df.groupBy('col').agg(avg('val'))` | `df %>% group_by(col) %>% summarize(mean = mean(val))` | `dt[, .(mean = mean(val)), by = col]` |
| `df.groupby('col').size()` | `df %>% group_by(col) %>% summarize(n = n())` | `dt[, .N, by = col]` |
| `df.groupby('col').agg({'val': ['mean', 'sum']})` | `df %>% group_by(col) %>% summarize(mean = mean(val), sum = sum(val))` | `dt[, .(mean = mean(val), sum = sum(val)), by = col]` |
| `df.groupby(['a', 'b'])['c'].sum()` | `df %>% group_by(a, b) %>% summarize(sum = sum(c))` | `dt[, .(sum = sum(c)), by = .(a, b)]` |

### Joins

| Python/PySpark | R dplyr | R data.table |
|----------------|---------|--------------|
| `pd.merge(left, right, on='id')` | `left_join(left, right, by = 'id')` | `merge(left, right, by = 'id')` or `right[left]` |
| `left.join(right, on='id', how='left')` | `left_join(left, right, by = 'id')` | `merge(left, right, by = 'id', all.x = TRUE)` |
| `left.join(right, on='id', how='inner')` | `inner_join(left, right, by = 'id')` | `merge(left, right, by = 'id')` |
| `left.join(right, on='id', how='outer')` | `full_join(left, right, by = 'id')` | `merge(left, right, by = 'id', all = TRUE)` |
| `left.join(right, on='id', how='right')` | `right_join(left, right, by = 'id')` | `merge(left, right, by = 'id', all.y = TRUE)` |
| `pd.merge(left, right, left_on='a', right_on='b')` | `left_join(left, right, by = c('a' = 'b'))` | `merge(left, right, by.x = 'a', by.y = 'b')` |

### Window Functions

| Python/PySpark | R dplyr | R data.table |
|----------------|---------|--------------|
| `df.groupby('id')['val'].cumsum()` | `df %>% group_by(id) %>% mutate(cumsum = cumsum(val))` | `dt[, cumsum := cumsum(val), by = id]` |
| `df.withColumn('rank', rank().over(window))` | `df %>% group_by(grp) %>% mutate(rank = row_number())` | `dt[, rank := frank(val), by = grp]` |
| `window = Window.partitionBy('id').orderBy('val')` + `df.withColumn('rank', rank().over(window))` | `df %>% group_by(id) %>% mutate(rank = row_number(val))` | `dt[order(val), rank := 1:.N, by = id]` |
| `df.withColumn('lag', lag('val').over(window))` | `df %>% group_by(id) %>% mutate(lag = lag(val))` | `dt[, lag := shift(val, 1), by = id]` |
| `df.withColumn('lead', lead('val').over(window))` | `df %>% group_by(id) %>% mutate(lead = lead(val))` | `dt[, lead := shift(val, -1, type='lead'), by = id]` |

### Reshaping

| Python/PySpark | R dplyr | R data.table |
|----------------|---------|--------------|
| `df.pivot_table(values='val', index='id', columns='cat')` | `df %>% pivot_wider(names_from = cat, values_from = val)` | `dcast(dt, id ~ cat, value.var = 'val')` |
| `df.melt(id_vars=['id'], value_vars=['a', 'b'])` | `df %>% pivot_longer(cols = c(a, b), names_to = 'var', values_to = 'val')` | `melt(dt, id.vars = 'id', measure.vars = c('a', 'b'))` |
| `pd.crosstab(df['a'], df['b'])` | `df %>% count(a, b) %>% pivot_wider(names_from = b, values_from = n)` | `dcast(dt, a ~ b, fun.aggregate = length)` |

### Unique/Distinct

| Python/PySpark | R dplyr | R data.table |
|----------------|---------|--------------|
| `df['col'].unique()` | `unique(df$col)` or `distinct(df, col)` | `unique(dt$col)` |
| `df.drop_duplicates()` | `distinct(df)` | `unique(dt)` |
| `df.drop_duplicates(subset=['col'])` | `distinct(df, col, .keep_all = TRUE)` | `unique(dt, by = 'col')` |
| `df.select('col').distinct()` | `df %>% distinct(col)` | `unique(dt[, .(col)])` |

### String Operations

| Python/PySpark | R dplyr (stringr) | R data.table |
|----------------|-------------------|--------------|
| `df['col'].str.lower()` | `mutate(df, col = str_to_lower(col))` | `dt[, col := tolower(col)]` |
| `df['col'].str.upper()` | `mutate(df, col = str_to_upper(col))` | `dt[, col := toupper(col)]` |
| `df['col'].str.contains('pattern')` | `filter(df, str_detect(col, 'pattern'))` | `dt[grepl('pattern', col)]` |
| `df['col'].str.replace('old', 'new')` | `mutate(df, col = str_replace(col, 'old', 'new'))` | `dt[, col := gsub('old', 'new', col)]` |
| `df['col'].str.split(',')` | `mutate(df, col = str_split(col, ','))` | `dt[, col := strsplit(col, ',')]` |
| `df['col'].str.strip()` | `mutate(df, col = str_trim(col))` | `dt[, col := trimws(col)]` |

### Missing Data

| Python/PySpark | R dplyr (tidyr) | R data.table |
|----------------|-----------------|--------------|
| `df.isna()` | `is.na(df)` | `is.na(dt)` |
| `df.dropna()` | `drop_na(df)` or `na.omit(df)` | `na.omit(dt)` |
| `df.fillna(0)` | `replace_na(df, list(col = 0))` | `dt[is.na(col), col := 0]` |
| `df['col'].fillna(method='ffill')` | `fill(df, col, .direction = 'down')` | `dt[, col := nafill(col, type='locf')]` |
| `df.dropna(subset=['col'])` | `drop_na(df, col)` | `dt[!is.na(col)]` |

### Date/Time Operations

| Python/PySpark | R dplyr (lubridate) | R data.table |
|----------------|---------------------|--------------|
| `pd.to_datetime(df['col'])` | `mutate(df, col = as.Date(col))` | `dt[, col := as.Date(col)]` |
| `df['date'].dt.year` | `mutate(df, year = year(date))` | `dt[, year := year(date)]` |
| `df['date'].dt.month` | `mutate(df, month = month(date))` | `dt[, month := month(date)]` |
| `df['date'].dt.day` | `mutate(df, day = day(date))` | `dt[, day := mday(date)]` |
| `df['date'].dt.dayofweek` | `mutate(df, wday = wday(date))` | `dt[, wday := wday(date)]` |
| `df['date'].dt.floor('D')` | `mutate(df, date = floor_date(date, 'day'))` | `dt[, date := as.Date(date)]` |

### Combining DataFrames

| Python/PySpark | R dplyr | R data.table |
|----------------|---------|--------------|
| `pd.concat([df1, df2])` | `bind_rows(df1, df2)` | `rbindlist(list(dt1, dt2))` |
| `pd.concat([df1, df2], axis=1)` | `bind_cols(df1, df2)` | `cbind(dt1, dt2)` |
| `df1.union(df2)` | `bind_rows(df1, df2)` | `rbindlist(list(dt1, dt2))` |

### Database Operations

| Python/PySpark | R dplyr (dbplyr) | Notes |
|----------------|------------------|-------|
| `spark.read.jdbc(url, table)` | `tbl(con, 'table')` | Lazy evaluation |
| `df.write.jdbc(url, table, mode='overwrite')` | `dbWriteTable(con, 'table', df)` | Executes immediately |
| `df.filter(...).select(...).groupBy(...)` | `tbl(con, 'table') %>% filter(...) %>% select(...) %>% group_by(...)` | Translates to SQL |
| `df.collect()` | `collect()` | Brings data to memory |
| `df.explain()` | `show_query()` or `explain()` | Shows SQL/execution plan |

---

## Session 6: Resources & Next Steps (20 min)

### Essential Documentation

1. **Tidyverse**: https://www.tidyverse.org/
   - dplyr: https://dplyr.tidyverse.org/
   - tidyr: https://tidyr.tidyverse.org/
   - readr: https://readr.tidyverse.org/

2. **data.table**: https://rdatatable.gitlab.io/data.table/
   - Vignettes: https://cran.r-project.org/web/packages/data.table/vignettes/

3. **dbplyr**: https://dbplyr.tidyverse.org/
   - Database backends: https://db.rstudio.com/

4. **R for Data Science (free book)**: https://r4ds.hadley.nz/

### Recommended Learning Path

**Week 1-2: Master the Basics**
- [ ] Complete dplyr tutorial
- [ ] Practice all 5 main verbs daily
- [ ] Convert 3-5 of your Python scripts to R

**Week 3-4: Performance Optimization**
- [ ] Learn data.table syntax
- [ ] Benchmark dplyr vs data.table on your datasets
- [ ] Identify which tool fits your use cases

**Week 5-6: Database Integration**
- [ ] Set up DB2 connection
- [ ] Convert SQL queries to dbplyr
- [ ] Build end-to-end pipeline

**Week 7-8: Advanced Topics**
- [ ] Explore purrr for functional programming
- [ ] Learn ggplot2 for visualization
- [ ] Study R Markdown for reports

### Quick Reference: Common Functions

#### dplyr "Cheat Verbs"
```r
# The 5 main verbs
filter()      # Subset rows
select()      # Subset columns
mutate()      # Create/modify columns
arrange()     # Sort rows
summarize()   # Aggregate data

# Grouping
group_by()    # Group data
ungroup()     # Remove grouping

# Joining
left_join()   # Left join
inner_join()  # Inner join
full_join()   # Full outer join
semi_join()   # Filter left based on right match
anti_join()   # Filter left based on NO right match

# Window functions (use with mutate after group_by)
row_number()  # Ranking 1,2,3,4...
dense_rank()  # Ranking 1,2,2,3...
lag()         # Previous row value
lead()        # Next row value
cumsum()      # Cumulative sum
```

#### data.table Special Symbols
```r
.N            # Count rows
.SD           # Subset of Data (current group)
.SDcols       # Columns to include in .SD
.I            # Row numbers
.BY           # Current group values
:=            # Assign by reference
%between%     # x >= left & x <= right
%chin%        # Fast %in% for characters
%like%        # Pattern matching
```

### Common Gotchas & Troubleshooting

```r
# 1. Forgetting to ungroup()
df %>%
  group_by(category) %>%
  mutate(avg = mean(value)) %>%
  ungroup()  # IMPORTANT! Future operations may behave unexpectedly

# 2. NSE (Non-Standard Evaluation) in functions
# Use {{ }} (curly-curly) for column names in functions
my_summary <- function(data, group_col, value_col) {
  data %>%
    group_by({{ group_col }}) %>%
    summarize(mean = mean({{ value_col }}))
}

# 3. data.table modifies by reference
dt[, new_col := value]  # dt is MODIFIED (no copy)
dt2 <- dt              # dt2 points to SAME data
# Use copy() for independent copy
dt2 <- copy(dt)

# 4. dbplyr lazy evaluation
query <- tbl(con, "table") %>% filter(x > 5)
# Nothing happens until:
result <- collect(query)  # NOW it executes

# 5. tibble vs data.frame differences
tibble(x = 1:3, y = x * 2)     # Works! (uses x)
data.frame(x = 1:3, y = x * 2) # Error! (x not found)
```

### Practice Exercises

#### Exercise 1: Data Transformation
```r
# Create sample data
library(nycflights13)  # Install if needed
data("flights")

# Task: Find the top 5 airlines by average arrival delay
# Filter out cancelled flights and delays < 0
# Your code here:
```

#### Exercise 2: data.table Performance
```r
# Create large dataset
set.seed(123)
large_dt <- data.table(
  id = sample(1:10000, 1e6, replace = TRUE),
  category = sample(LETTERS[1:5], 1e6, replace = TRUE),
  value = rnorm(1e6)
)

# Task: Calculate mean and sd of value by id and category
# Compare performance: base R, dplyr, data.table
# Your code here:
```

#### Exercise 3: Database Workflow
```r
# Task: Create a customer segmentation report
# 1. Connect to database
# 2. Load customers and transactions tables
# 3. Join and aggregate to find RFM metrics
# 4. Create customer segments
# 5. Visualize results
# Your code here:
```

### Next Steps: Going Further

1. **Explore tidymodels** for machine learning in R
2. **Learn sparklyr** for distributed computing (Spark + R)
3. **Master ggplot2** for publication-quality visualizations
4. **Study R Markdown/Quarto** for reproducible reports
5. **Dive into Shiny** for interactive web applications

---

## Appendix: Installation & Setup Scripts

### Setting up IBM DB2 JDBC Driver

```bash
# Download IBM DB2 JDBC driver
# Visit: https://www.ibm.com/support/pages/db2-jdbc-driver-versions-and-downloads

# Extract and note the path to db2jcc4.jar
# Example: /opt/ibm/db2/java/db2jcc4.jar

# Add to R environment
# In ~/.Renviron:
DB2_JDBC_PATH="/opt/ibm/db2/java/db2jcc4.jar"
```

### RStudio Configuration

```r
# Set default options in ~/.Rprofile
options(
  repos = c(CRAN = "https://cloud.r-project.org"),
  stringsAsFactors = FALSE,
  max.print = 100,
  scipen = 999  # Disable scientific notation
)

# Load commonly used packages
if (interactive()) {
  suppressMessages({
    library(tidyverse)
    library(data.table)
  })
}

# Custom prompt
if (interactive()) {
  cat("\n\nWelcome to R! Tidyverse and data.table loaded.\n\n")
}
```

### Performance Benchmarking Template

```r
library(microbenchmark)
library(ggplot2)

# Benchmark template
benchmark_results <- microbenchmark(
  base_r = {
    # Base R code
  },
  dplyr = {
    # dplyr code
  },
  data.table = {
    # data.table code
  },
  times = 10
)

# Visualize
autoplot(benchmark_results)

# Summary
summary(benchmark_results)
```

---

## Summary & Key Takeaways

### Core Concepts
✅ **dplyr**: Readable, intuitive, excellent for data exploration and collaboration  
✅ **data.table**: Ultra-fast, memory-efficient, best for large datasets and production  
✅ **dbplyr**: Work with databases using dplyr syntax, let DB do the heavy lifting  
✅ **Pipe %>%**: Chain operations for readable data pipelines  
✅ **Lazy evaluation**: dbplyr doesn't execute until collect()  

### When to Use What
- **Small-medium data (<1M rows)**: dplyr for readability
- **Large data (>1M rows)**: data.table for speed
- **Database data**: dbplyr for efficiency
- **Best of both**: dtplyr (dplyr syntax + data.table speed)

### Python → R Mental Model
- `df.method()` → `df %>% verb()`
- `df.groupby().agg()` → `df %>% group_by() %>% summarize()`
- Method chaining → Pipe operator
- PySpark SQL → dbplyr

### You're Ready When...
- [ ] You can translate any pandas operation to dplyr
- [ ] You understand when to use data.table vs dplyr
- [ ] You can connect to databases and write efficient queries
- [ ] You can benchmark and optimize R code
- [ ] You think in pipes and verbs

**Happy coding! 🚀**

---

*Training session created: 2025  
Last updated: October 2025  
Author: For Python/PySpark users transitioning to R*
