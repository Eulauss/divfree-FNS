from divfree.common.io_utils import ModelCheckpoint

ckpt = "runs_test_new_functions/divfree2d-nomass-k=4-lstsq-saveall/check_points/divfree2d_M=12.npz"

model = ModelCheckpoint(ckpt)

print(model.a)
