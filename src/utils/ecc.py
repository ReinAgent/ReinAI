import math

def get_inverse_element(value, max_value):
    for i in range(1, max_value):
        if (i * value) % max_value == 1:
            return i
    return -1


def gcd_x_y(x, y):
    if y == 0:
        return x
    else:
        return gcd_x_y(y, x % y)
    

def calculate_p_q(x1,y1,x2,y2, a, p):
    flag = 1  
    if x1 == x2 and y1 == y2:
        member = 3 * (x1 ** 2) + a 
        denominator = 2 * y1 
    else:
        member = y2 - y1
        denominator = x2 - x1 
        if member* denominator < 0:
            flag = 0
            member = abs(member)
            denominator = abs(denominator)
    
    gcd_value = gcd_x_y(member, denominator)
    member = int(member / gcd_value)
    denominator = int(denominator / gcd_value)
    # 求分母的逆元    
    inverse_value = get_inverse_element(denominator, p)
    k = (member * inverse_value)
    if flag == 0:
        k = -k
    k = k % p
    # 计算x3,y3
    x3 = (k ** 2 - x1 - x2) % p
    y3 = (k * (x1 - x3) - y1) % p
    # print("%d<=====>%d" % (x3, y3))
    return [x3,y3]
    

def get_order(x0, y0, a, b, p):
    # 计算-p
    x1 = x0
    y1 = (-1 * y0) % p
    temp_x = x0
    temp_y = y0
    n = 1
    while True:
        n += 1
        p_value = calculate_p_q(temp_x,temp_y, x0, y0, a, p)
        if p_value[0] == x1 and p_value[1] == y1:
            return n+1
            
        temp_x = p_value[0]
        temp_y = p_value[1]

    # print("%d-%d-%d-%d" % (x0,y0,x1,y1))


def get_x0_y0_x1_y1(x0, a, b, p):
    y0 = -1
    for i in range(0,p):
        if i ** 2 % p == (x0**3 + a*x0 + b) % p:
            y0 = i
            break
    if y0 == -1:
        return False
    x1 = x0
    y1 = -1 * y0 % p
    return [x0,y0,x1,y1]


def draw_graph(a,b,p):
    x_y = []
    for i in range(p):
        x_y.append(["-" for i in range(p)])
    
    for i in range(p):
        value = get_x0_y0_x1_y1(i, a, b, p)
        if value != False:
            x0 = value[0]
            y0 = value[1]
            x1 = value[2]
            y1 = value[3]
            # print("%d-%d-%d-%d" % (x0,y0,x1,y1))
            x_y[x0][y0] = 1
            x_y[x1][y1] = 1

    for j in range(p):
        if p-1-j >= 10:
            print(p-1-j, end=" ")
        else:
             print(p-1-j, end="  ")
        for i in range(p):
            print(x_y[i][p-j-1], end="  ")
        print()
    print("   ",end="")
    for i in range(p):
        if i >= 10:
            print(i, end=" ")
        else:
            print(i, end="  ")
        
    print()


def calculate_np(G_x, G_y, private_key, a, p):
    temp_x = G_x
    temp_y = G_y
    while private_key != 1:
        p_value = calculate_p_q(temp_x,temp_y, G_x, G_y, a, p)
        temp_x = p_value[0]
        temp_y = p_value[1]
        private_key -= 1
    return p_value

def ecc_encrypt_and_decrypt():
    while True:
        a = int(input("Please input the parameter a of the elliptic curve: "))
        b = int(input("Please input the parameter b of the elliptic curve: "))
        p = int(input("Please input the parameter p of the elliptic curve (p should be a prime number): "))
        
        if (4*(a**3) + 27*(b**2)) % p == 0:
            print("The selected elliptic curve cannot be used for encryption, please choose again.\n")
        else:
            break
    # Output the scatter plot of the elliptic curve
    draw_graph(a, b, p)
    print("Select a point as the generator G from the graph above.")
    G_x = int(input("Input the x-coordinate of the selected generator G_x: "))
    G_y = int(input("Input the y-coordinate of the selected generator G_y: "))
    # Get the order of the elliptic curve
    n = get_order(G_x, G_y, a, b, p)
    # Get the private key, ensuring key < the order of the elliptic curve (n)
    private_key = int(input("Input the private key (<%d): " % n))
    # Calculate the public key nG
    Q = calculate_np(G_x, G_y, private_key, a, p)
    print("==================Generated Public Key {a=%d, b=%d, p=%d, order=%d, G(%d,%d), Q(%d,%d)}======" % 
          (a, b, p, n, G_x, G_y, Q[0], Q[1]))
    
    # Encryption starts
    k = int(input("Provide an integer (<%d): " % n))
    k_G = calculate_np(G_x, G_y, k, a, p)  # Calculate kG
    k_Q = calculate_np(Q[0], Q[1], k, a, p)  # Calculate kQ
    plain_text = int(input("Input the plaintext to be encrypted: "))
    cipher_text = plain_text * k_Q[0]  # Calculate the product of the plaintext and the x-coordinate of kQ
    # The ciphertext is
    C = [k_G[0], k_G[1], cipher_text]
    print("The ciphertext is: {(%d, %d), %d}" % (C[0], C[1], C[2]))
    
    # Decryption
    # Calculate private_key * kG
    decrypto_text = calculate_np(C[0], C[1], private_key, a, p)
    
    inverse_value = get_inverse_element(decrypto_text[0], p)
    m = C[2] * inverse_value % p
    print("The decrypted plaintext is: %d" % m)    