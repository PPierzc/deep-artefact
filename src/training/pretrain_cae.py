

def pretrain_cae(
    x,
    cae,
    batch_size=256,
    epochs=200,
    optimizer='adam',
    save_dir=None,
    **kwargs
):
    cae.compile(optimizer=optimizer, loss='mse')

    history = cae.fit(
        x,
        x,
        batch_size=batch_size,
        epochs=epochs,
        verbose=1,
        **kwargs
    )

    cae.save(save_dir + 'pretrain_cae_model.h5')

    print('Pretrained weights are saved to %s/pretrain_cae_model.h5' % save_dir)
    return history
