-- Question 1
USE online_clothing_store;
INSERT INTO customers(firstname,lastname,mobile_number,email,username,address,state,points,password,registeredAt)
VALUE ('Todsavad','Tangtortan','6-885-211-3463','st2312@ait.asia',
'34y34g4','Mr. Eduardo Grant, Arline  1234, Lancaster - 0062, Mali','Nevada',500,'12yt12gettrrtsb',now());
SELECT * FROM customers where customer_id = 52;


-- Question 2
USE online_clothing_store;
INSERT INTO products(vendor_id,product_name,discount,price,quantity,category_id,description,availability_period,createdAt)
VALUE (10,'Stwwihu','70.3223','2400.34', '50' , '5','tvAA0HtnbB5skRAfuSvahUOh2iQ5H','2007-07-27 20:07:21',now());
SELECT * FROM products where product_id = 104;

-- Question 3 Insert a new order
USE online_clothing_store;
INSERT INTO orders(invoice_number, customer_id,vendor_id, quantity, total, payment_method, ordered_date, points_awarded, createdAt)
VALUE (6969, 5, 49, 9, 3934, 'online', now(), 99, now());
SELECT * FROM orders WHERE order_id = 301;

-- Question 4 
USE online_clothing_store;
UPDATE products
SET price = 3200, updatedAt = now()
WHERE product_id = 1;
SELECT * FROM products WHERE product_id = 1;

-- Question 5
USE online_clothing_store;
UPDATE customers
SET address = 'Mrs. Irene Clifton, Balfe 435, Escondido - 2578, Ukraine', updatedAt = now()
WHERE customer_id = 1;
SELECT * FROM customers WHERE customer_id = 1;

-- Question 6
USE online_clothing_store;
UPDATE product_reviews
SET product_rating = 4,description = 'This shirt is very fit for me. I love this shirt!!!', updatedAt = now()
WHERE review_id = 5;
SELECT * FROM product_reviews WHERE review_id = 5;

-- Question 7
USE online_clothing_store;
UPDATE delivery_status
SET delivery_status = 'delivered'
WHERE delivery_id = 6;
SELECT * FROM delivery_status WHERE delivery_id = 6;

-- Question 8
DELETE 
FROM products WHERE product_id=101;
SELECT * FROM products WHERE product_id=101;

-- Question 9
DELETE 
FROM customers WHERE customer_id = 52;
SELECT * FROM customers WHERE customer_id = 52;

-- Question 10
DELETE 
FROM product_reviews WHERE review_id = 517;
SELECT * FROM product_reviews ORDER BY review_id DESC;

-- Question 11
SELECT storename, ratings FROM vendors
ORDER BY ratings DESC;

-- Question 12
SELECT CONCAT(c.firstName, '', COALESCE(c.middleName, ' '), ' ', c.lastName) as Customername,storename,c.state 
FROM customers c 
INNER JOIN vendors v ON c.state = v.state 
ORDER BY state;
