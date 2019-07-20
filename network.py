from keras.layers import Conv2D, Conv2DTranspose, MaxPool2D, Concatenate, Input
import keras
import apply_nms
import numpy as np

PRED_DOWNSCALE_FACTORS = (8,4,2,1)
GAMMA = [1,1,2,4]
NUM_BOXES_PER_SCALE = 3
BOX_SIZE_BINS = [1]
BOX_IDX = [0]
g_idx = 0
while len(BOX_SIZE_BINS) < NUM_BOXES_PER_SCALE * len(PRED_DOWNSCALE_FACTORS):
    gamma_idx = len(BOX_SIZE_BINS) // (len(GAMMA)-1)
    box_size = BOX_SIZE_BINS[g_idx] + GAMMA[gamma_idx]
    box_idx = gamma_idx*(NUM_BOXES_PER_SCALE+1) + (len(BOX_SIZE_BINS) % (len(GAMMA)-1))
    BOX_IDX.append(box_idx)
    BOX_SIZE_BINS.append(box_size)
    g_idx += 1
BOX_INDEX = dict(zip(BOX_SIZE_BINS, BOX_IDX))
SCALE_BINS_ON_BOX_SIZE_BINS = [NUM_BOXES_PER_SCALE * (s + 1) \
                               for s in range(len(GAMMA))]
BOX_SIZE_BINS_NPY = np.array(BOX_SIZE_BINS)
BOXES = np.reshape(BOX_SIZE_BINS_NPY, (4, 3))
BOXES = BOXES[::-1]

UPSAMPLE_SCALES = [4,3,2,1]

RGB_MEANS = np.array([104.008, 116.669, 122.675]).reshape(1,1,1,3)

def upsample2x(img):
    img = img.copy()
    shape = img.shape
    rows, cols = np.indices((shape[1],shape[2]))
    # newimindices = np.indices((shape[1],shape[2]))
    res = np.zeros((shape[0],shape[1]*2,shape[2]*2,shape[3]))
    res[:,:,:,3] = 1
    res[:,rows*2,cols*2,:] = img[:,rows,cols,:]
    return res

def upsample2xXtimes(img,times):
    res = img.copy()
    for _ in range(times):
        res = upsample2x(res)
    return res

def upsample_predictions(preds):
    for i,pred in enumerate(preds):
        preds[i] = upsample2xXtimes(pred,UPSAMPLE_SCALES[i])
    return preds

def get_box_and_dot_maps(pred, thresh):
    # NMS on the multi-scale outputs
    nms_out, h = box_NMS(pred, thresh)
    return nms_out, h
    
def box_NMS(predictions, thresh):
    Scores = []
    Boxes = []
    for k in range(len(BOXES)):
        scores = np.max(predictions[k], axis=0)
        boxes = np.argmax(predictions[k], axis=0)
        # index the boxes with BOXES to get h_map and w_map (both are the same for us)
        mask = (boxes<3) # removing Z
        boxes = (boxes+1) * mask
        scores = (scores * mask) # + 100 # added 100 since we take logsoftmax and it's negative!! 
    
        boxes = (boxes==1)*BOXES[k][0] + (boxes==2)*BOXES[k][1] + (boxes==3)*BOXES[k][2]
        Scores.append(scores)
        Boxes.append(boxes)

    x, y, h, w, scores = apply_nms.apply_nms(Scores, Boxes, Boxes, 0.5, thresh=thresh)
    
    nms_out = np.zeros((predictions[0].shape[1], predictions[0].shape[2])) # since predictions[0] is of size 4 x H x W
    box_out = np.zeros((predictions[0].shape[1], predictions[0].shape[2])) # since predictions[0] is of size 4 x H x W
    for (xx, yy, hh) in zip(x, y, h):
        nms_out[yy, xx] = 1
        box_out[yy, xx] = hh
    
    assert(np.count_nonzero(nms_out) == len(x))

    return nms_out, box_out

def preprocess_image(img):
    img -= RGB_MEANS
    return img


def pred_for_one_image(model,image,threshold=0.3):
    image = np.expand_dims(image,0).astype(np.float32)
    image = preprocess_image(image)
    preds = model.predict(image)
    s1,s2,s3,s4 = upsample_predictions([*preds])
    preds = [s1[0],s2[0],s3[0],s4[0]]
    # the nms function i use expects the input to be in
    # a channels first format, akin to pytorch.
    preds = [np.transpose(p,(2,0,1)) for p in preds]
    w,h = get_box_and_dot_maps(preds,threshold)
    box_locs = w > 0
    return box_locs,w,h
    

def build_LSCCNN_model():
    # vgg block 1
    conv1_1 = Conv2D(64,(3,3),padding='same',activation='relu',name='conv1_1')
    conv1_2 = Conv2D(64,(3,3),padding='same',activation='relu',name='conv1_2')
    pool1 = MaxPool2D()
    
    #vgg block 2
    conv2_1 = Conv2D(128,(3,3),padding='same',activation='relu',name='conv2_1')
    conv2_2 = Conv2D(128,(3,3),padding='same',activation='relu',name='conv2_2')
    pool2 = MaxPool2D()

    #vgg block 3
    conv3_1 = Conv2D(256,(3,3),padding='same',activation='relu',name='conv3_1')
    conv3_2 = Conv2D(256,(3,3),padding='same',activation='relu',name='conv3_2')
    conv3_3 = Conv2D(256,(3,3),padding='same',activation='relu',name='conv3_3')
    pool3 = MaxPool2D()

    #vgg block 4
    conv4_1 = Conv2D(512,(3,3),padding='same',activation='relu',name='conv4_1')
    conv4_2 = Conv2D(512,(3,3),padding='same',activation='relu',name='conv4_2')
    conv4_3 = Conv2D(512,(3,3),padding='same',activation='relu',name='conv4_3')
    pool4 = MaxPool2D()

    #vgg block 5
    conv5_1 = Conv2D(512,(3,3),padding='same',activation='relu',name='conv5_1')
    conv5_2 = Conv2D(512,(3,3),padding='same',activation='relu',name='conv5_2')
    conv5_3 = Conv2D(512,(3,3),padding='same',activation='relu',name='conv5_3')

    # scale 1
    convA_1 = Conv2D(256,(3,3),padding='same',activation='relu',name='convA_1')
    convA_2 = Conv2D(128,(3,3),padding='same',activation='relu',name='convA_2')
    convA_3 = Conv2D(64,(3,3),padding='same',activation='relu',name='convA_3')
    convA_4 = Conv2D(32,(3,3),padding='same',activation='relu',name='convA_4')
    convA_5 = Conv2D(4,(3,3),padding='same',name='convA_5')

    # scale 2
    convB_1 = Conv2D(256,(3,3),padding='same',activation='relu',name='convB_1')
    convB_2 = Conv2D(128,(3,3),padding='same',activation='relu',name='convB_2')
    convB_3 = Conv2D(64,(3,3),padding='same',activation='relu',name='convB_3')
    convB_4 = Conv2D(32,(3,3),padding='same',activation='relu',name='convB_4')
    convB_5 = Conv2D(4,(3,3),padding='same',name='convB_5')

    # scale 3
    convC_1 = Conv2D(256,(3,3),padding='same',activation='relu',name='convC_1')
    convC_2 = Conv2D(128,(3,3),padding='same',activation='relu',name='convC_2')
    convC_3 = Conv2D(64,(3,3),padding='same',activation='relu',name='convC_3')
    convC_4 = Conv2D(32,(3,3),padding='same',activation='relu',name='convC_4')
    convC_5 = Conv2D(4,(3,3),padding='same',name='convC_5')

    # scale 4
    convD_1 = Conv2D(256,(3,3),padding='same',activation='relu',name='convD_1')
    convD_2 = Conv2D(128,(3,3),padding='same',activation='relu',name='convD_2')
    convD_3 = Conv2D(64,(3,3),padding='same',activation='relu',name='convD_3')
    convD_4 = Conv2D(32,(3,3),padding='same',activation='relu',name='convD_4')
    convD_5 = Conv2D(4,(3,3),padding='same',name='convD_5')

    #mfr 1
    conv_before_transpose_1 = Conv2D(256,(3,3),padding='same',activation='relu',name='conv_before_transpose_1')
    transpose_1 = Conv2DTranspose(256,3,strides=2,padding='same',activation='relu',output_padding=1,name='transpose_1')
    conv_after_transpose_1_1  = Conv2D(256,3,padding='same',activation='relu',name='conv_after_transpose_1_1')

    #mfr 2
    transpose_2 = Conv2DTranspose(256,3,strides=2,padding='same',activation='relu',output_padding=1,name='transpose_2')
    conv_after_transpose_2_1 = Conv2D(128,3,padding='same',activation='relu',name='conv_after_transpose_2_1')

    #mfr 3
    transpose_3 = Conv2DTranspose(256,3,strides=4,output_padding=1,name='transpose_3')
    conv_after_transpose_3_1 = Conv2D(128,3,padding='same',activation='relu',name='conv_after_transpose_3_1')

    #mfr 4
    transpose_4_1_a = Conv2DTranspose(256,3,strides=4,output_padding=1,name='transpose_4_1_a')
    transpose_4_1_b = Conv2DTranspose(256,3,strides=2,padding='same',activation='relu',output_padding=1,name='transpose_4_1_b')
    conv_after_transpose_4_1 = Conv2D(64,3,padding='same',activation='relu',name='conv_after_transpose_4_1')

    transpose_4_2 = Conv2DTranspose(256,3,strides=4,output_padding=1,name='transpose_4_2')
    conv_after_transpose_4_2 = Conv2D(64,3,padding='same',activation='relu',name='conv_after_transpose_4_2')

    transpose_4_3 = Conv2DTranspose(128,3,strides=2,padding='same',activation='relu',output_padding=1,name='transpose_4_3')
    conv_after_transpose_4_3 = Conv2D(64,3,padding='same',activation='relu',name='conv_after_transpose_4_3')

    conv_middle_1 = Conv2D(512,3,padding='same',activation='relu',name='conv_middle_1')
    conv_middle_2 = Conv2D(512,3,padding='same',activation='relu',name='conv_middle_2')
    conv_middle_3 = Conv2D(512,3,padding='same',activation='relu',name='conv_middle_3')
    conv_mid_4 = Conv2D(256,3,padding='same',activation='relu',name='conv_mid_4')

    conv_lowest_1 = Conv2D(256,3,padding='same',activation='relu',name='conv_lowest_1')
    conv_lowest_2 = Conv2D(256,3,padding='same',activation='relu',name='conv_lowest_2')
    conv_lowest_3 = Conv2D(256,3,padding='same',activation='relu',name='conv_lowest_3')
    conv_lowest_4 = Conv2D(128,3,padding='same',activation='relu',name='conv_lowest_4')

    conv_scale1_1 = Conv2D(128,3,padding='same',activation='relu',name='conv_scale1_1')
    conv_scale1_2 = Conv2D(128,3,padding='same',activation='relu',name='conv_scale1_2')
    conv_scale1_3 = Conv2D(64,3,padding='same',activation='relu',name='conv_scale1_3')

    input_image = Input(shape=(None,None,3))
    
    ######################## Stage 1 ###########################
    main_out_block1 = conv1_2(conv1_1(input_image))
    main_out_pool1 = pool1(main_out_block1)

    main_out_block2 = conv2_2(conv2_1(main_out_pool1))
    main_out_pool2 = pool2(main_out_block2)

    main_out_block3 = conv3_3(conv3_2(conv3_1(main_out_pool2)))
    main_out_pool3 = pool3(main_out_block3)

    main_out_block4 = conv4_3(conv4_2(conv4_1(main_out_pool3)))
    main_out_pool4 = pool4(main_out_block4)

    main_out_block5 = conv_before_transpose_1(conv5_3(conv5_2(conv5_1(main_out_pool4))))

    main_out_rest = convA_5(convA_4(convA_3(convA_2(convA_1(main_out_block5)))))
    
    ######################## Stage 2 ###########################

    sub1_out_conv1 = conv_mid_4(conv_middle_3(conv_middle_2(conv_middle_1(main_out_pool3))))
    sub1_transpose = transpose_1(main_out_block5)
    sub1_after_transpose_1 = conv_after_transpose_1_1(sub1_transpose)

    sub1_concat = Concatenate(axis=3)([sub1_out_conv1,sub1_after_transpose_1])
    sub1_out_rest = convB_5(convB_4(convB_3(convB_2(convB_1(sub1_concat)))))

    ######################## Stage 3 ###########################

    sub2_out_conv1 = conv_lowest_4(conv_lowest_3(conv_lowest_2(conv_lowest_1(main_out_pool2))))
    sub2_transpose = transpose_2(sub1_out_conv1)
    sub2_after_transpose_1 = conv_after_transpose_2_1(sub2_transpose)

    sub3_transpose = transpose_3(main_out_block5)
    sub3_after_transpose_1 = conv_after_transpose_3_1(sub3_transpose)

    sub2_concat = Concatenate(axis=3)([sub2_out_conv1,sub2_after_transpose_1,sub3_after_transpose_1])
    sub2_out_rest = convC_5(convC_4(convC_3(convC_2(convC_1(sub2_concat)))))

    ######################## Stage 4 ###########################

    sub4_out_conv1 = conv_scale1_3(conv_scale1_2(conv_scale1_1(main_out_pool1)))

    # TDF 1 
    tdf_4_1_a = transpose_4_1_a(main_out_block5)
    tdf_4_1_b = transpose_4_1_b(tdf_4_1_a)
    after_tdf_4_1 = conv_after_transpose_4_1(tdf_4_1_b)

    # TDF 2 
    tdf_4_2 = transpose_4_2(sub1_out_conv1)
    after_tdf_4_2 = conv_after_transpose_4_2(tdf_4_2)

    #tdf 3
    tdf_4_3 = transpose_4_3(sub2_out_conv1)
    after_tdf_4_3 = conv_after_transpose_4_3(tdf_4_3)

    sub4_concat = Concatenate(axis=3)([sub4_out_conv1,after_tdf_4_1,after_tdf_4_2,after_tdf_4_3])
    sub4_out_rest = convD_5(convD_4(convD_3(convD_2(convD_1(sub4_concat)))))

    lsccnnmodel = keras.Model(inputs=input_image,outputs=[main_out_rest,sub1_out_rest,sub2_out_rest,sub4_out_rest])

    return lsccnnmodel

if __name__ == "__main__":
    mdl = build_LSCCNN_model()
    mdl.summary(line_length=128)
    # dummyimage = np.random.randn(1,640,480,3)
    dummyimage = np.zeros((1,640,480,3))
    a,b,c,d = mdl.predict(dummyimage)
    print(a.shape)
    print(b.shape)
    print(c.shape)
    print(d.shape)
    newa,newb,newc,newd = upsample_predictions([a,b,c,d])
    print(newa.shape)
    print(newb.shape)
    print(newc.shape)
    print(newd.shape)
    preds = [newa[0],newb[0],newc[0],newd[0]]
    preds = [np.transpose(p,(2,0,1)) for p in preds]
    nms_out, h = get_box_and_dot_maps(preds,0.9)
    print(nms_out.shape)
    print(nms_out)
    print(h)