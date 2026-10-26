-- Q1: WHERE then HAVING
SELECT customer_id, SUM(order_amount) as total_amount
FROM orders
WHERE status = 'completed' AND order_amount > 500
GROUP BY customer_id
HAVING SUM(order_amount) > 8000;

-- Q2: Three-table join (Revenue by city and product category)
SELECT c.city, p.category, SUM(o.order_amount) as revenue
FROM orders o
JOIN customers c ON o.customer_id = c.customer_id
JOIN products p ON o.product_id = p.product_id
WHERE o.status = 'completed'
GROUP BY c.city, p.category;

-- Q3: Scalar subquery (Products priced above average)
SELECT product_name, price_inr
FROM products
WHERE price_inr > (SELECT AVG(price_inr) FROM products);

-- Q4: LEFT JOIN (Customers with no completed order)
SELECT c.customer_id
FROM customers c
LEFT JOIN orders o ON c.customer_id = o.customer_id AND o.status = 'completed'
WHERE o.order_id IS NULL;

-- Q5: Common table expression (Above average spenders)
WITH CustomerTotals AS (
    SELECT customer_id, SUM(order_amount) as total_spend
    FROM orders
    WHERE status = 'completed'
    GROUP BY customer_id
),
AverageSpend AS (
    SELECT AVG(total_spend) as avg_total
    FROM CustomerTotals
)
SELECT ct.customer_id, ct.total_spend
FROM CustomerTotals ct, AverageSpend a
WHERE ct.total_spend > a.avg_total;

-- Q6: Ranking window functions
SELECT c.customer_id, c.city, SUM(o.order_amount) as total_spend,
       RANK() OVER (PARTITION BY c.city ORDER BY SUM(o.order_amount) DESC) as rnk,
       DENSE_RANK() OVER (PARTITION BY c.city ORDER BY SUM(o.order_amount) DESC) as dense_rnk,
       ROW_NUMBER() OVER (PARTITION BY c.city ORDER BY SUM(o.order_amount) DESC) as row_num
FROM orders o
JOIN customers c ON o.customer_id = c.customer_id
WHERE o.status = 'completed'
GROUP BY c.customer_id, c.city;

-- Q7: Running total
SELECT SUBSTR(order_ts, 1, 10) as day, 
       SUM(order_amount) as daily_revenue,
       SUM(SUM(order_amount)) OVER (ORDER BY SUBSTR(order_ts, 1, 10)) as cumulative_revenue
FROM orders
WHERE status = 'completed'
GROUP BY SUBSTR(order_ts, 1, 10);