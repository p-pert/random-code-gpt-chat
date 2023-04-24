
        if model == "Dense":
                print("Data preprocessing DENSE")
                print("Before preprocessig")
                print(x[0])
                if(onehot == 1):
                    # One hot encoded representation to be used with embedding.
                    # Event is represented with a 2048 dim array where =1 elements are in 
                    # positions of the array corresponding to turned on pmts numbers
                    nevents = x.shape[0]
                    # prepare the np array to store x as one hot encoded
                    ohe = np.zeros((nevents, 2048), np.uint16)
                    # extract indices of positive x elements.
                    idx = np.where(x > 0)
                    # idx[0] is an array containing the row indices of the positive x elements 
                    # and idx[1] is an array containing the col idxs of the same elements
                    # x[idx] = turned on pmts numbers
                    ohe[idx[0], x[idx].astype(int)] = 1
                    # Since it's one-hot pack as binary to reduce size of npy file
                    # ATTENTION! Use this packaging only if ohe axis 1 is a multiple of 8! (like 2048)
                    x = np.packbits(ohe, axis=1)
                    print("After one-hot encoding and binary packaging")
                    print(x[0])
                    # Test
                    #xtest = np.unpackbits(x, axis=1)
                    #print(xtest[0])
                    #assert(np.allclose(ohe, xtest))

                elif(1): # bunching
                    # first one hot encode in this case too
                    nevents = x.shape[0]
                    ohe = np.zeros((nevents, 2048), np.uint16)
                    idx = np.where(x > 0)
                    ohe[idx[0], x[idx].astype(int)] = 1
                    # now the bunching in doubles:  
                    import struct

                    PMTS = 2048
                    bunch_size = 32

                    # x is a numpy array of shape (Nevents, PMTS), x_new will have shape (Nevents, PMTS//bunch_size)

                    x_new = np.zeros((nevents, PMTS//bunch_size), np.double)

                    for x,x_new in zip(x[0:1],x_new[0:1]):

                        k = 0
                        while(k < PMTS):

                            i = 0 # this int will become a double in the end
                            for j,bit in enumerate(x[k : k+bunch_size]):
                                if(bit == 1): # we will add a bit in the mantissa
                                    #print("found a bit=1 at id =", k+j)
                                    i |= 0x0000000000100000   # =  0 | 00000000000 | 0000000000000000000000000000000100000000000000000000 = sign | exp | mantissa   # we populate the mantissa most significant 32 bits
                                i = i << 1

                                # a couple prints to make sure it is working:
                                #print(i >> 1)
                                #byt_i = (i >> 1).to_bytes(8, 'big') # convert the integer to a bytes object ready to be packed into a double
                                #print('0x'+byt_i.hex() )

                            i = i >> 1    # undo the last shift of the loop
                            i |= 0x3FF0000000000000   # =  0 | 01111111111 | 0000000000000000000000000000000000000000000000000000 = sign | exp | mantissa  # we set sign and exp to our liking
                            byt = i.to_bytes(8, 'big') # convert the integer to a bytes object ready to be packed into a double
                            d = struct.unpack('>d', byt)[0] # unpack it as a double value
                            #print(" ")
                            #print("Bunch that starts with k = ", k, "completed. Here's the integer in binary: ")
                            #print("0x"+byt.hex(), " and it's value as a double = ", d)
                            #print(" ")

                            x_new[k//bunch_size] = d

                            k = k + bunch_size

                    x = x_new
                    print("After bunching, bunch size =", bunch_size)
                    print(x[0])

                else:
                    x = x /2048.
                    print("After normalization")
                    print(x[0])
