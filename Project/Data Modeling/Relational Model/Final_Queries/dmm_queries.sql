-- Question 1
use online_clothing_store;
select firstname,count(o.customer_id) as count 
from customers c 
inner join orders o on 
c.customer_id = o.customer_id
where c.points >500
group by c.customer_id
having count>10;

-- Question 2
USE online_clothing_store;
SELECT v.storename, MAX(p.price) AS MaximumPrice, p.product_name
FROM vendors v
INNER JOIN products p ON p.vendor_id = v.vendor_id
GROUP BY v.storename;

-- Question 3
USE online_clothing_store;
SELECT v.vendor_id,v.storename, sum(o.total) AS '1st Quarter Total Order'
FROM orders o
INNER JOIN vendors v ON o.vendor_id = v.vendor_id
WHERE o.ordered_date BETWEEN '2020-01-01' AND '2020-03-31' 
GROUP BY v.vendor_id ;

-- Question 4
USE online_clothing_store;
SELECT v.vendor_id,v.storename, sum(o.total) AS '2nd QuarterTotal Order'
FROM orders o
INNER JOIN vendors v ON o.vendor_id = v.vendor_id
WHERE o.ordered_date BETWEEN '2020-04-01' AND '2020-06-30' 
GROUP BY v.vendor_id ;

-- Question 5
USE online_clothing_store;
SELECT v.vendor_id,v.storename, sum(o.total) AS '3rd QuarterTotal Order'
FROM orders o
INNER JOIN vendors v ON o.vendor_id = v.vendor_id
WHERE o.ordered_date BETWEEN '2020-07-01' AND '2020-09-30' 
GROUP BY v.vendor_id ;

-- Question 6
USE online_clothing_store;
SELECT v.vendor_id,v.storename, sum(o.total) AS '4th QuarterTotal Order'
FROM orders o
INNER JOIN vendors v ON o.vendor_id = v.vendor_id
WHERE o.ordered_date BETWEEN '2020-10-01' AND '2020-12-31' 
GROUP BY v.vendor_id ;

-- Question 7
USE online_clothing_store;
SELECT v.vendor_id,v.storename, sum(o.total) AS 'Total Revenue'
FROM orders o
INNER JOIN vendors v ON o.vendor_id = v.vendor_id
GROUP BY v.vendor_id
ORDER BY v.vendor_id;

-- Question 8
USE online_clothing_store;
SELECT v.vendor_id,v.storename, v.state, count(o.product_id) AS 'NoofOrder'
FROM vendors v 
INNER JOIN products p ON p.vendor_id = v.vendor_id
INNER JOIN orders_products o ON o.product_id = p.product_id
GROUP BY v.vendor_id
ORDER BY NoofOrder DESC;

-- Question 9
USE online_clothing_store;
SELECT v.vendor_id,v.storename, sum(o.quantity) AS 'Total Orders'
FROM orders o
INNER JOIN vendors v ON o.vendor_id = v.vendor_id
WHERE o.ordered_date BETWEEN '2022-01-01' AND '2022-04-30' 
GROUP BY v.vendor_id;

-- Question 10
USE online_clothing_store;
SELECT c.category_id, c.category_name, SUM(o.product_id) AS NoofProduct FROM categories c
INNER JOIN products p ON p.category_id = c.category_id
INNER JOIN orders_products o ON o.product_id = p.product_id
GROUP BY c.category_id
ORDER BY NoofProduct DESC;

-- Question 11
select state , count(o.customer_id) as number_of_online_orders
from customers c 
inner join orders o on o.customer_id=c.customer_id 
where payment_method ='online' 
group by state 
order by number_of_online_orders desc 
limit 5;

-- Question 12
SELECT 
    state,storename, COUNT(op.order_id) AS number_of_orders , ratings
FROM
    vendors v
        INNER JOIN
    products p ON v.vendor_id = p.vendor_id
        INNER JOIN
    orders_products op ON op.product_id = p.product_id
WHERE
    ratings >= 3
GROUP BY v.vendor_id
HAVING number_of_orders > 11
ORDER BY number_of_orders DESC
limit 5;

-- Question 13
SELECT 
    state,storename, COUNT(op.order_id) AS number_of_orders , ratings
FROM
    vendors v
        INNER JOIN
    products p ON v.vendor_id = p.vendor_id
        INNER JOIN
    orders_products op ON op.product_id = p.product_id
WHERE
    ratings <= 2
GROUP BY v.vendor_id
HAVING number_of_orders > 11
ORDER BY number_of_orders DESC
limit 5;

-- Question 14
USE online_clothing_store;
SELECT v.vendor_id,v.firstname,v.storename,sum(p.quantity)
FROM vendors v
INNER JOIN products p ON p.vendor_id = v.vendor_id
GROUP BY v.vendor_id
ORDER BY v.vendor_id;

-- Question 15
select  CONCAT(c.firstName, ' ', COALESCE(c.middleName, ' '), ' ',   c.lastName) as customername,storename ,c.state
 from customers c
 inner join vendors v on 
 c.state = v.state
 order by state;
 
-- Question 16
SELECT v.vendor_id, v.storename, v.state, COUNT(ds.delivery_status) AS CountReturn 
FROM vendors v  
INNER JOIN products p ON p.vendor_id = v.vendor_id 
INNER JOIN orders_products op ON op.product_id = p.product_id 
INNER JOIN delivery_status ds ON ds.order_id = op.order_id 
WHERE ds.delivery_status = 'returned' 
GROUP BY v.state ORDER BY CountReturn DESC LIMIT 1;

-- Question 17
Select storename, ratings from vendors where ratings <= 2 Order by ratings desc;

-- Question 18
SELECT 
    c.username,
    CONCAT(c.firstName,
            ' ',
            COALESCE(c.middleName, ' '),
            ' ',
            c.lastName) AS fullname,
    COUNT(ds.order_id) AS number_of_cancelled_orders
FROM
    delivery_status ds
        INNER JOIN
    orders o ON o.order_id = ds.order_id
        INNER JOIN
    customers c ON c.customer_id = o.customer_id
WHERE
    delivery_status = 'cancelled'
GROUP BY c.customer_id
HAVING number_of_cancelled_orders > 3
ORDER BY number_of_cancelled_orders;

-- Question 19
SELECT c.customer_id,c.firstname,sum(o.total)as totalspent from customers c
  inner join  orders o on o.customer_id = c.customer_id 
  where createdAt between '2021-01-01' AND '2021-12-31'
  group by c.customer_id
  having totalspent>50000
  order by totalspent desc;

-- Question 20
SELECT p.product_name, avg(pr.product_rating) as average_rating from product_reviews pr
inner join products p on p.product_id = pr.product_id
group by p.product_name;

-- Question 21
Select returned_reason, count(returned_reason) as total 
from delivery_status where returned_date is not null 
group by returned_reason limit 1;

-- Question 22
SELECT 
    v.storename,
    COUNT(ds.order_id) AS number_of_orders_placed,
    ratings,
    v.state
FROM
    delivery_status ds
        INNER JOIN
    orders o ON o.order_id = ds.order_id
        INNER JOIN
    vendors v ON v.vendor_id = o.vendor_id
WHERE
    ratings <= 3
        AND delivery_status = 'returned'
GROUP BY v.storename
HAVING number_of_orders_placed >= 2
ORDER BY ratings 
limit 5;

-- Question 23
SELECT
(SELECT count(*) FROM orders WHERE CAST(orders.ordered_date as TIME) BETWEEN '06:00' AND '12:00') as '6AM to 12PM',
(SELECT count(*) FROM orders WHERE CAST(orders.ordered_date as TIME) BETWEEN '12:00' AND '18:00') as '12PM to 17PM',
(SELECT count(*) FROM orders WHERE CAST(orders.ordered_date as TIME) BETWEEN '18:00' AND '24:00') as '18PM - 12AM',
(SELECT count(*) FROM orders WHERE CAST(orders.ordered_date as TIME) BETWEEN '00:00' AND '06:00') as '00AM - 06AM'