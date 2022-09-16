SELECT * FROM sns.user;

-- Task 1 --
SELECT posterId, message,createdAt FROM sns.user_post
ORDER BY createdAt DESC ;

-- Task 2 --
SELECT id, firstName, middleName, lastName FROM sns.user
WHERE middleName IS NULL
ORDER BY lastName ASC ;

-- Task 3 --
SELECT id, firstName, LastName, gender,registeredAt
FROM sns.user 
WHERE registeredAt BETWEEN '2022-08-14' AND '2022-08-26'
ORDER BY registeredAt;

-- SELECT * FROM sns.user
-- WHERE registeredAt > DATE_ADD(NOW(), INTERVAL -14 DAY)

-- Task 4 --
SELECT * FROM sns.user_post 
WHERE wallId = 1 AND createdAt < '2021-08-01' 
ORDER BY createdAt DESC;

-- Task 5 --
SELECT id, targetId, updatedAt
FROM sns.user_friend
WHERE status = 'rejected' and sourceId = 66;

-- Task 6 --
SELECT sourceId ,status, COUNT(*) AS NoOfFriend
FROM sns.user_friend
WHERE status = 'accepted'
GROUP BY sourceId 
ORDER BY NoOfFriend DESC;

-- Task 7 --
SELECT id, COUNT(timestampdiff(DAY, createdAt, now()))
AS CountDAY, updatedAt
FROM sns.user_friend 
GROUP BY id
ORDER BY updatedAt DESC LIMIT 1;

SELECT targetid, timestampDIFF(DAY,updatedAt,now()) 
FROM sns.user_friend 
WHERE sourceId = 32 AND updatedAt IS NOT NULL 
ORDER BY updatedAt DESC LIMIT 1;

-- Task 8 --
SELECT *
FROM sns.group
WHERE status != 'blocked'
ORDER BY createdAt DESC;

-- Task 9 --
SELECT groupID, status, COUNT(*) AS NoOfMember 
FROM sns.group_member
WHERE status = 'approved'
GROUP BY groupId 
ORDER BY NoOfMember ASC;

-- Task 10 --
SELECT userId, status, COUNT(*) AS NoOfMember 
FROM sns.group_member
WHERE status = 'approved'
GROUP BY userId 
ORDER BY NoOfMember DESC;

-- Task 11 --
SELECT groupID, status, COUNT(*) AS NoOfGroup
FROM sns.group_member
WHERE status = 'approved'
GROUP BY groupId 
ORDER BY NoOfGroup DESC LIMIT 10;

-- SELECT user.id,user.username,COUNT(groupId) 
-- FROM group_member 
-- INNER JOIN `user` ON group_member.userId = user.id 
-- WHERE group_member.status = 'approved'
-- GROUP BY user.id 
-- ORDER BY COUNT(groupId) DESC LIMIT 10;

-- Task 12 --
SELECT id,title,createdAt, COUNT(*) AS MaxMember
FROM sns.group 
WHERE createdAt BETWEEN '2022-08-01' AND '2022-08-31'
GROUP BY title 
ORDER BY MaxMember DESC;

-- SELECT groupId ,COUNT(message) AS MaxMember
-- FROM sns.group_post 
-- WHERE status = 'approved' AND createdAt > 2022-08-01
-- GROUP BY groupId 
-- ORDER BY MaxMember DESC;

-- Task 13 --
SELECT *
FROM sns.user_post 
WHERE message NOT LIKE '%shit%' 
OR message NOT LIKE '%bad%'
OR message NOT LIKE '%hell%' 
OR message NOT LIKE '%crap%' 
OR message NOT LIKE '%fuck%';


