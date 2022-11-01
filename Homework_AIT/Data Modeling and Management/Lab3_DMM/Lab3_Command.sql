-- Retrieve all --
SELECT * FROM sns.user; 

-- Retrieve specific -- 
-- SELECT id, firstName, LastName, gender FROM sns.user;

-- Retrieve specific --
-- SELECT DISTINCT gender FROM sns.user;
-- Unique value

-- Only female --
-- SELECT id, firstName, LastName, gender 
-- FROM sns.user 
-- WHERE gender = 'f';

-- Only Null --
-- SELECT id, firstName, LastName, gender 
-- FROM sns.user 
-- WHERE gender IS NULL;

-- Not Null --
-- SELECT id, firstName, LastName, gender 
-- FROM sns.user 
-- WHERE gender IS NOT NULL;

-- Range --
-- SELECT id, firstName, LastName, gender 
-- FROM sns.user 
-- WHERE gender = 'm' 
-- AND id BETWEEN 10 AND 100;

-- IN Operator -- 
-- SELECT id, firstName, LastName, gender 
-- FROM sns.user 
-- WHERE firstName IN ('Jennifer','Ardelia','Chase');


-- Exercise 1 --
-- start name begin with 'S'
-- SELECT id, firstName, LastName, gender 
-- FROM sns.user 
-- WHERE firstName LIKE 'S%';

-- not provide their gender and telephone
-- SELECT id, firstName, LastName, gender 
-- FROM sns.user 
-- WHERE gender IS NULL AND mobile IS NULL;

-- Last 7 Days --
-- SELECT id, firstName, LastName, gender,registeredAt
-- FROM sns.user 
-- WHERE registeredAt BETWEEN '2022-08-19' AND '2022-08-25'
-- ORDER BY registeredAt;

-- LIKE and WILDCARD --
-- Prefix --
-- SELECT id, firstName
-- FROM sns.user 
-- WHERE firstName LIKE '%en%';

-- Postfix --
-- SELECT id, firstName
-- FROM sns.user 
-- WHERE firstName LIKE 'Jen%';

-- Sorting --
-- _ mean only one character
-- SELECT id, firstName
-- FROM sns.user 
-- WHERE firstName LIKE '_er%%'; 

-- Sorting Data --
-- default is ASC
-- SELECT firstName, middleName, LastName 
-- FROM sns.user 
-- ORDER BY firstName;

-- Exercise2 --

-- begin with 'Jo' descending order using lastname
-- SELECT firstName, middleName, LastName 
-- FROM sns.user 
-- WHERE firstName LIKE 'Jo%'
-- ORDER BY LastName DESC;

-- female firstname OR is not contain ,bel,
-- SELECT firstName, LastName, gender 
-- FROM sns.user 
-- WHERE (gender = 'f') 
-- OR (gender IS NULL AND firstName LIKE '%bel%');

-- Aggregation --
-- SELECT COUNT(*) AS NoOfPost FROM sns.user_post;

-- SELECT * FROM sns.user_post;

-- SELECT posterId, COUNT(*) AS NoOfPost 
-- FROM sns.user_post GROUP BY posterId;

-- SELECT AVG(timestampdiff(MONTH, registeredAt, now()))
-- AS AVGDuration, id FROM sns.user GROUP BY id;

-- 08/26/2022 = 05/26/2022 = 3 month / 90 days / ... hours / second and so on

-- Exercise 3 --
-- whihch gender has more averange membering duration
SELECT gender, AVG(timestampdiff(MONTH, registeredAt, now()))
AS AVGMembership
FROM sns.user 
WHERE gender IS NOT NULL
GROUP BY gender;

-- Limiting Output -- 
-- SELECT * FROM sns.user LIMIT 3;
-- SELECT * FROM sns.user LIMIT 2,10;

-- Exercsie 4 --
-- last 4 group on the basis of group member
-- SELECT * FROM sns.group_member;
SELECT groupId , count(*) AS Members 
FROM sns.group_member 
GROUP BY groupId 
ORDER BY Members DESC LIMIT 4;

-- 3 male and 3 female user who have the longest membership duration
-- male
SELECT firstName, gender, registeredAt
FROM sns.user
WHERE gender = 'm' 
ORDER BY registeredAt DESC LIMIT 3;

-- female
SELECT firstName, gender, registeredAt
FROM sns.user
WHERE gender = 'f' 
ORDER BY registeredAt DESC LIMIT 3;
