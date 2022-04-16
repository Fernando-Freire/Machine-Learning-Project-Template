import pandas as pd

from scripts.models.word_embeddings.gensim import Gensim


def test_pre_process_data():

    sentences = [
        "Eu não tenho 30 tangerinas! prato bom chocolate bulbasauro",
        "A dona aranha subiu pela parede molhada",
    ]

    tokens = Gensim._pre_process_data(data=pd.Series(sentences))
    tokens_list = tokens.tolist()
    assert len(tokens_list[0]) == 8
    assert len(tokens_list[1]) == 6


def test_get_features():
    sentences = [
        "Eu não tenho 30 tangerinas! prato bom chocolate bulbasauro",
        "A dona aranha subiu pela parede molhada",
    ]
    tokens = Gensim._pre_process_data(pd.Series(sentences))

    result = Gensim._get_features(
        prepared_train_data=tokens, word_model_type="FastText"
    )
    assert type(result) is pd.Series
    print(result)
    # test field type str
    assert result.size == 2

    result = Gensim._get_features(
        prepared_train_data=tokens, word_model_type="Word2Vec"
    )
    assert type(result) is pd.Series
    # test field type str
    assert result.size == 2

    result = Gensim._get_features(prepared_train_data=tokens, word_model_type="Doc2Vec")
    assert type(result) is pd.Series
    # test TaggedDocument
    assert result.size == 2


# def test_Word2Vec_codify():
#     sentences = [
#         "Eu não tenho 30 tangerinas! prato bom chocolate bulbasauro",
#         "Eu tenho 30 tangerinas! prato bom chocolate bulbasauro"
#     ]
#     gensim = Gensim(word_model_type="Word2Vec")
#     tokens = gensim._pre_process_data(sentences)

#     result = gensim._get_features(
#         prepared_train_data=tokens, word_model_type="Word2Vec"
#     )
#     model = gensim._export_model()
#     model.build_vocab(result)
#     model.train(
#         result,
#         total_examples=model.corpus_count,
#         epochs=model.epochs,
#     )
#     codified = gensim._codify(
#           result.tolist(),
#           word_model_type="Word2Vec",
#            model=model,dimensions=100)

#     sentences2 = [
#         "A dona aranha subiu pela parede molhada tadinha"
#     ]
#     tokens2 = gensim._pre_process_data(sentences2)

#     result2 = gensim._get_features(
#         prepared_train_data=tokens2, word_model_type="Word2Vec"
#     )

#     codified2 = gensim._codify(result2.tolist(),
#            word_model_type="Word2Vec",
#            model=model,dimensions=100)
#     assert codified == codified2


# def test_Doc2Vec_codify():
#     sentences = [
#         "Eu não tenho 30 tangerinas! prato bom chocolate bulbasauro",
#         "A dona aranha subiu pela parede molhada",
#     ]
#     gensim = Gensim(word_model_type="Doc2Vec")
#     tokens = gensim._pre_process_data(sentences)

#     result = gensim._get_features(prepared_train_data=tokens,
#           word_model_type="Doc2Vec")
#     model = gensim._export_model()
#     model.build_vocab(result)
#     model.train(
#         result,
#         total_examples=model.corpus_count,
#         epochs=model.epochs,
#     )
#     codified = gensim._codify(result, word_model_type="Doc2Vec", model=model)

#     sentences2 = [
#         "Eu não tenho 30 tangerinas! prato bom chocolate bulbasauro",
#         "A dona aranha subiu pela parede molhada tadinha",
#     ]
#     tokens2 = gensim._pre_process_data(sentences2)

#     result2 = gensim._get_features(
#         prepared_train_data=tokens2, word_model_type="Doc2Vec"
#     )

#     codified2 = gensim._codify(result2, word_model_type="Doc2Vec", model=model)
#     assert codified == codified2


# def test_FastText_codify():
#     sentences = [
#         "Eu não tenho 30 tangerinas! prato bom chocolate bulbasauro",
#         "A dona aranha subiu pela parede molhada",
#     ]
#     gensim = Gensim(word_model_type="FastText")
#     tokens = gensim._pre_process_data(sentences)

#     result = gensim._get_features(
#         prepared_train_data=tokens, word_model_type="FastText"
#     )
#     model = gensim._export_model()
#     model.build_vocab(result)
#     model.train(
#         result,
#         total_examples=model.corpus_count,
#         epochs=model.epochs,
#     )
#     codified = gensim._codify(result, word_model_type="FastText", model=model)

#     sentences2 = [
#         "Eu não tenho 30 tangerinas! prato bom chocolate bulbasauro",
#         "A dona aranha subiu pela parede molhada tadinha",
#     ]
#     tokens2 = gensim._pre_process_data(sentences2)

#     result2 = gensim._get_features(
#         prepared_train_data=tokens2, word_model_type="FastText"
#     )

#     codified2 = gensim._codify(result2, word_model_type="FastText", model=model)
#     assert codified != codified2


# def test_get_feature_matrix():
#     sentences = [
#         "Eu não tenho 30 tangerinas! prato bom chocolate bulbasauro",
#         "A dona aranha subiu pela parede molhada",
#     ]
#     gensim = Gensim(word_model_type="FastText")
#     tokens = gensim._pre_process_data(sentences)

#     result = gensim._get_features(
#         prepared_train_data=tokens, word_model_type="FastText"
#     )
#     model = gensim._export_model()
#     model.build_vocab(result)
#     model.train(
#         result,
#         total_examples=model.corpus_count,
#         epochs=model.epochs,
#     )
#     feature_matrix = gensim._get_feature_matrix(
#         features=result, word_model_type="FastText", model=model
#     )

#     sentences2 = [
#         "Eu não tenho 30 tangerinas! prato bom chocolate bulbasauro",
#         "A dona aranha subiu pela parede molhada tadinha",
#     ]
#     tokens2 = gensim._pre_process_data(sentences2)

#     result2 = gensim._get_features(
#         prepared_train_data=tokens2, word_model_type="FastText"
#     )
#     feature_matrix2 = gensim._get_feature_matrix(
#         features=result2, word_model_type="FastText", model=model
#     )

#     assert feature_matrix != feature_matrix2
