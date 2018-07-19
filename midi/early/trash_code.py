def _update_Spr(Gp, Gr, Rpr, Spr_shape, I, B, t):
    # F Reconstruction
    F = np.zeros(Spr_shape)
    F += np.linalg.multi_dot((Gp.T, Rpr[0], Gr))
    Gk = Gp * I[:, t][:, np.newaxis]
    Ik = np.reshape(I[:, t], (len(I[:, t]), 1))
    bk = np.reshape(B[t, :], (1, len(B[t, :])))
    F += 2*np.linalg.multi_dot((Gk.T, Ik, bk))
    # # C construction
    # C = np.zeros(Spr_shape)
    # Gks = []
    # bks = []
    # for l in range(len(Rpr)):
    #     C += np.linalg.multi_dot((Gp.T, Rpr[l], Gr))
    # for tk in range(I.shape[1]):
    #     Gk = Gp * I[:, tk][:, np.newaxis]
    #     Gks.append(Gk)
    #     Ik = np.reshape(I[:, tk], (len(I[:, tk]), 1))
    #     bk = np.reshape(B[tk, :], (1, len(B[tk, :])))
    #     bks.append(bk)
    #     C += 2*np.linalg.multi_dot((Gk.T, Ik, bk))
    # print(C.shape)

    # A construction
    A = Gp.T.dot(Gp)
    C = Gk.T.dot(Gk)
    # for tk in range(I.shape[1]):
    #     A.append(2*Gks[tk].T.dot(Gks[tk]))
    #print(A)

    # B construction
    B = Gr.T.dot(Gr)
    D = bk.T.dot(bk)

    pinv_C = np.linalg.pinv(C)
    pinv_B = np.linalg.pinv(B)
    A_new = pinv_C.dot(A)
    B_new = D.dot(pinv_B)
    C_new = pinv_C.dot(F.dot(pinv_B))
    return solve_sylvester(A_new, B_new, C_new)
    # for tk in range(I.shape[1]):
    #     B.append(bks[tk].T.dot(bks[tk]))
    #print(B)
    # print("sto cominciando gd")
    # S = np.zeros(Spr_shape)
    # mu = 1e-7
    # for i in range(100):
    #     S_old = S
    #     print("siamo all'iterzione ", i)
    #
    #     print(_M(A=A, B=B, X=S))
    #     #print(C)
    #     grad = (_M(A=A, B=B, X=S) - C)
    #     S = S + mu * (grad)
    #     #print(S)
    #     if np.linalg.norm(S-S_old) < 1e-3:
    #         break
    # print(S)
    # print("numero di iterazioni fatte %d, valore a convergenza %4f" %(i, np.linalg.norm(S-S_old)))
    # # gss = GeneralizedSylvesterSolver(max_iter=500, verbose=0)
    # gss.fit(A, B, C)

    #return S


def _update_B(Gp, Spr, I):
    B = np.zeros((I.shape[1], Spr[0].shape[1]))
    for t in range(I.shape[1]):
        y = np.reshape(I[:, t], (len(I[:, t]), 1))
        X = Gp.dot(Spr[0]) * y
        ls = LinearRegression()
        ls.fit(X, y)
        B[t, :] = ls.coef_
    return B

    
def _iterative_gradient_solver(A, B, C, tol=1e-3, max_iter=500, verbose=0,
                                random_state=0):
    return None
#
#     Xs = [random_state.randn(C.shape[0], C.shape[1])]
#     iterates_error = []
#     objective_error = []
#     mu = (1/2*np.sum([np.linalg.norm(Aj)*np.linalg.norm(Bj)
#                       for Aj, Bj in zip(A, B)]))-1e-5
#     for _iter in range(max_iter):
#         AXB = _M(A=A, B=B, X=Xs[-1].copy())
#         gradients = []
#         for i in range(len(A)):
#             gradients.append(np.linalg.multi_dot((A[i].T, C - AXB, B[i].T)))
#         if(np.any(np.isnan(np.array(gradients)))):
#             print("bbbbbbbbbbbbbbbbbbbbbbbbbbbbb")
#             break
#         X = np.sum(np.array([Xs[-1] + mu*grad
#                              for grad in gradients]), axis=0)/len(A)
#         if(np.any(np.isnan(X))):
#             print("aaaaaaaaaaaaaaaaaaaaaaaaaacle")
#             break
#         Xs.append(X)
#         iterates_error.append(np.linalg.norm(Xs[-2] - Xs[-1]))
#         objective_error.append(np.linalg.norm(C - _M(A, B, Xs[-1])))
#
#         if objective_error[-1] < tol:
#             print("iteration %d" % _iter)
#             break
#     else:
#         print("The algorithm did not converge")
#     return X
#
#
def _iterative_gradient_smaller_solver(A, B, F, tol=1e-3, max_iter=200,
                                        verbose=0, random_state=0):
    return None
#
#     Xs = [np.zeros(C.shape)]
#     iterates_error = []
#     objective_error = []
#     print(2*np.sum([np.linalg.norm(Aj)*np.linalg.norm(Bj)
#                       for Aj, Bj in zip(A, B)]))
#     mu = (1/(2*np.sum([np.linalg.norm(Aj)*np.linalg.norm(Bj)
#                       for Aj, Bj in zip(A, B)])))-1e-7
#     print(mu)
#     a = A[0]
#     c = A[1]
#     b = B[0]
#     d = B[1]
#     for _iter in range(max_iter):
#         grad = F - a.dot(Xs[-1]).dot(b) - c.dot(Xs[-1]).dot(d)
#         print(grad)
#         X_1 = Xs[-1] + mu * a.T.dot(grad).dot(b.T)
#         X_2 = Xs[-1] + mu * c.T.dot(grad).dot(d.T)
#         Xs.append(np.array((X_1 + X_2)/2))
#         # print(Xs[-2])
#         # print(Xs[-1])
#         diff = Xs[-2] - Xs[-1]
#         # print(1e-4 < 1e-3)
#         # print(np.linalg.norm(diff))
#         # print(tol)
#         iterates_error.append(np.linalg.norm(diff)/F.shape[0])
#         #print(iterates_error[-1])
#
#         objective_error.append(np.linalg.norm(F - _M(A=A, B=B, X=Xs[-1]))/F.shape[0])
#         # print(objective_error[-1])
#         # print(float(objective_error[-1]) < float(tol))
#         if objective_error[-1] < tol:
#             print("iteration %d" % _iter)
#             break
#     else:
#         print("The algorithm did not converge")
#     return Xs[-1]tr
