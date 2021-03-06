/*
 * Copyright (c) 2020, Peter Abeles. All Rights Reserved.
 *
 * This file is part of BoofCV (http://boofcv.org).
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package boofcv.alg.feature.detect.intensity.impl;

import boofcv.struct.image.GrayU8;

import javax.annotation.Generated;

/**
 * <p>
 * Contains logic for detecting fast corners. Pixels are sampled such that they can eliminate the most
 * number of possible corners, reducing the number of samples required.
 * </p>
 *
 * <p>DO NOT MODIFY. Automatically generated code created by GenerateImplFastCorner</p>
 *
 * @author Peter Abeles
 */
@Generated("boofcv.alg.feature.detect.intensity.impl.GenerateImplFastCorner")
public class ImplFastCorner11_U8 extends ImplFastHelper_U8 {
	public ImplFastCorner11_U8(int pixelTol) {
		super(pixelTol);
	}

	/**
	 * @return 1 = positive corner, 0 = no corner, -1 = negative corner
	 */
	@Override public final int checkPixel( int index ) {
		setThreshold(index);

		if ((data[index+offsets[0]]& 0xFF) > upper) {
			return function2(index);
		} else {
			return function3(index);
		}
	}

	@Override public FastCornerInterface<GrayU8> newInstance() {
		return new ImplFastCorner11_U8(tol);
	}

	public final int function2( int index ) {
		if ((data[index+offsets[5]]& 0xFF) > upper) {
			return function4(index);
		} else {
			return function5(index);
		}
	}

	public final int function3( int index ) {
		if ((data[index+offsets[5]]& 0xFF) < lower) {
			return function6(index);
		} else {
			return function7(index);
		}
	}

	public final int function4( int index ) {
		if ((data[index+offsets[1]]& 0xFF) > upper) {
			if ((data[index+offsets[2]]& 0xFF) > upper) {
				if ((data[index+offsets[3]]& 0xFF) > upper) {
					if ((data[index+offsets[4]]& 0xFF) > upper) {
						if ((data[index+offsets[6]]& 0xFF) > upper) {
							if ((data[index+offsets[7]]& 0xFF) > upper) {
								if ((data[index+offsets[8]]& 0xFF) > upper) {
									if ((data[index+offsets[9]]& 0xFF) > upper) {
										if ((data[index+offsets[10]]& 0xFF) > upper) {
											return 1;
										} else {
											if ((data[index+offsets[15]]& 0xFF) > upper) {
												return 1;
											} else {
												return 0;
											}
										}
									} else {
										if ((data[index+offsets[14]]& 0xFF) > upper) {
											if ((data[index+offsets[15]]& 0xFF) > upper) {
												return 1;
											} else {
												return 0;
											}
										} else {
											return 0;
										}
									}
								} else {
									if ((data[index+offsets[13]]& 0xFF) > upper) {
										if ((data[index+offsets[14]]& 0xFF) > upper) {
											if ((data[index+offsets[15]]& 0xFF) > upper) {
												return 1;
											} else {
												return 0;
											}
										} else {
											return 0;
										}
									} else {
										return 0;
									}
								}
							} else {
								if ((data[index+offsets[12]]& 0xFF) > upper) {
									if ((data[index+offsets[13]]& 0xFF) > upper) {
										if ((data[index+offsets[14]]& 0xFF) > upper) {
											if ((data[index+offsets[15]]& 0xFF) > upper) {
												return 1;
											} else {
												return 0;
											}
										} else {
											return 0;
										}
									} else {
										return 0;
									}
								} else {
									return 0;
								}
							}
						} else {
							if ((data[index+offsets[11]]& 0xFF) > upper) {
								if ((data[index+offsets[12]]& 0xFF) > upper) {
									if ((data[index+offsets[13]]& 0xFF) > upper) {
										if ((data[index+offsets[14]]& 0xFF) > upper) {
											if ((data[index+offsets[15]]& 0xFF) > upper) {
												return 1;
											} else {
												return 0;
											}
										} else {
											return 0;
										}
									} else {
										return 0;
									}
								} else {
									return 0;
								}
							} else {
								return 0;
							}
						}
					} else {
						if ((data[index+offsets[9]]& 0xFF) > upper) {
							if ((data[index+offsets[10]]& 0xFF) > upper) {
								if ((data[index+offsets[11]]& 0xFF) > upper) {
									if ((data[index+offsets[12]]& 0xFF) > upper) {
										if ((data[index+offsets[13]]& 0xFF) > upper) {
											if ((data[index+offsets[14]]& 0xFF) > upper) {
												if ((data[index+offsets[15]]& 0xFF) > upper) {
													return 1;
												} else {
													return 0;
												}
											} else {
												return 0;
											}
										} else {
											return 0;
										}
									} else {
										return 0;
									}
								} else {
									return 0;
								}
							} else {
								return 0;
							}
						} else {
							return 0;
						}
					}
				} else {
					if ((data[index+offsets[8]]& 0xFF) > upper) {
						if ((data[index+offsets[9]]& 0xFF) > upper) {
							if ((data[index+offsets[10]]& 0xFF) > upper) {
								if ((data[index+offsets[11]]& 0xFF) > upper) {
									if ((data[index+offsets[12]]& 0xFF) > upper) {
										if ((data[index+offsets[13]]& 0xFF) > upper) {
											if ((data[index+offsets[14]]& 0xFF) > upper) {
												if ((data[index+offsets[7]]& 0xFF) > upper) {
													if ((data[index+offsets[15]]& 0xFF) > upper) {
														return 1;
													} else {
														if ((data[index+offsets[4]]& 0xFF) > upper) {
															if ((data[index+offsets[6]]& 0xFF) > upper) {
																return 1;
															} else {
																return 0;
															}
														} else {
															return 0;
														}
													}
												} else {
													if ((data[index+offsets[15]]& 0xFF) > upper) {
														return 1;
													} else {
														return 0;
													}
												}
											} else {
												return 0;
											}
										} else {
											return 0;
										}
									} else {
										return 0;
									}
								} else {
									return 0;
								}
							} else {
								return 0;
							}
						} else {
							return 0;
						}
					} else {
						return 0;
					}
				}
			} else {
				if ((data[index+offsets[7]]& 0xFF) > upper) {
					if ((data[index+offsets[8]]& 0xFF) > upper) {
						if ((data[index+offsets[9]]& 0xFF) > upper) {
							if ((data[index+offsets[10]]& 0xFF) > upper) {
								if ((data[index+offsets[11]]& 0xFF) > upper) {
									if ((data[index+offsets[12]]& 0xFF) > upper) {
										if ((data[index+offsets[13]]& 0xFF) > upper) {
											if ((data[index+offsets[6]]& 0xFF) > upper) {
												if ((data[index+offsets[14]]& 0xFF) > upper) {
													if ((data[index+offsets[15]]& 0xFF) > upper) {
														return 1;
													} else {
														if ((data[index+offsets[4]]& 0xFF) > upper) {
															return 1;
														} else {
															return 0;
														}
													}
												} else {
													if ((data[index+offsets[3]]& 0xFF) > upper) {
														if ((data[index+offsets[4]]& 0xFF) > upper) {
															return 1;
														} else {
															return 0;
														}
													} else {
														return 0;
													}
												}
											} else {
												if ((data[index+offsets[14]]& 0xFF) > upper) {
													if ((data[index+offsets[15]]& 0xFF) > upper) {
														return 1;
													} else {
														return 0;
													}
												} else {
													return 0;
												}
											}
										} else {
											return 0;
										}
									} else {
										return 0;
									}
								} else {
									return 0;
								}
							} else {
								return 0;
							}
						} else {
							return 0;
						}
					} else {
						return 0;
					}
				} else {
					return 0;
				}
			}
		} else {
			if ((data[index+offsets[6]]& 0xFF) > upper) {
				if ((data[index+offsets[7]]& 0xFF) > upper) {
					if ((data[index+offsets[8]]& 0xFF) > upper) {
						if ((data[index+offsets[9]]& 0xFF) > upper) {
							if ((data[index+offsets[10]]& 0xFF) > upper) {
								if ((data[index+offsets[11]]& 0xFF) > upper) {
									if ((data[index+offsets[12]]& 0xFF) > upper) {
										if ((data[index+offsets[13]]& 0xFF) > upper) {
											if ((data[index+offsets[4]]& 0xFF) > upper) {
												if ((data[index+offsets[14]]& 0xFF) > upper) {
													return 1;
												} else {
													if ((data[index+offsets[3]]& 0xFF) > upper) {
														return 1;
													} else {
														return 0;
													}
												}
											} else {
												if ((data[index+offsets[14]]& 0xFF) > upper) {
													if ((data[index+offsets[15]]& 0xFF) > upper) {
														return 1;
													} else {
														return 0;
													}
												} else {
													return 0;
												}
											}
										} else {
											if ((data[index+offsets[2]]& 0xFF) > upper) {
												if ((data[index+offsets[3]]& 0xFF) > upper) {
													if ((data[index+offsets[4]]& 0xFF) > upper) {
														return 1;
													} else {
														return 0;
													}
												} else {
													return 0;
												}
											} else {
												return 0;
											}
										}
									} else {
										return 0;
									}
								} else {
									return 0;
								}
							} else {
								return 0;
							}
						} else {
							return 0;
						}
					} else {
						return 0;
					}
				} else {
					return 0;
				}
			} else {
				return 0;
			}
		}
	}

	public final int function5( int index ) {
		if ((data[index+offsets[10]]& 0xFF) > upper) {
			if ((data[index+offsets[11]]& 0xFF) > upper) {
				if ((data[index+offsets[12]]& 0xFF) > upper) {
					if ((data[index+offsets[13]]& 0xFF) > upper) {
						if ((data[index+offsets[14]]& 0xFF) > upper) {
							if ((data[index+offsets[15]]& 0xFF) > upper) {
								if ((data[index+offsets[1]]& 0xFF) > upper) {
									if ((data[index+offsets[9]]& 0xFF) > upper) {
										if ((data[index+offsets[2]]& 0xFF) > upper) {
											if ((data[index+offsets[8]]& 0xFF) > upper) {
												return 1;
											} else {
												if ((data[index+offsets[3]]& 0xFF) > upper) {
													return 1;
												} else {
													return 0;
												}
											}
										} else {
											if ((data[index+offsets[7]]& 0xFF) > upper) {
												if ((data[index+offsets[8]]& 0xFF) > upper) {
													return 1;
												} else {
													return 0;
												}
											} else {
												return 0;
											}
										}
									} else {
										if ((data[index+offsets[2]]& 0xFF) > upper) {
											if ((data[index+offsets[3]]& 0xFF) > upper) {
												if ((data[index+offsets[4]]& 0xFF) > upper) {
													return 1;
												} else {
													return 0;
												}
											} else {
												return 0;
											}
										} else {
											return 0;
										}
									}
								} else {
									if ((data[index+offsets[6]]& 0xFF) > upper) {
										if ((data[index+offsets[7]]& 0xFF) > upper) {
											if ((data[index+offsets[8]]& 0xFF) > upper) {
												if ((data[index+offsets[9]]& 0xFF) > upper) {
													return 1;
												} else {
													return 0;
												}
											} else {
												return 0;
											}
										} else {
											return 0;
										}
									} else {
										return 0;
									}
								}
							} else {
								return 0;
							}
						} else {
							return 0;
						}
					} else {
						return 0;
					}
				} else {
					return 0;
				}
			} else {
				return 0;
			}
		} else {
			if ((data[index+offsets[5]]& 0xFF) < lower) {
				if ((data[index+offsets[6]]& 0xFF) < lower) {
					if ((data[index+offsets[7]]& 0xFF) < lower) {
						if ((data[index+offsets[8]]& 0xFF) < lower) {
							if ((data[index+offsets[9]]& 0xFF) < lower) {
								if ((data[index+offsets[10]]& 0xFF) < lower) {
									if ((data[index+offsets[11]]& 0xFF) < lower) {
										if ((data[index+offsets[4]]& 0xFF) < lower) {
											if ((data[index+offsets[12]]& 0xFF) < lower) {
												if ((data[index+offsets[3]]& 0xFF) < lower) {
													if ((data[index+offsets[13]]& 0xFF) < lower) {
														return -1;
													} else {
														if ((data[index+offsets[2]]& 0xFF) < lower) {
															return -1;
														} else {
															return 0;
														}
													}
												} else {
													if ((data[index+offsets[13]]& 0xFF) < lower) {
														if ((data[index+offsets[14]]& 0xFF) < lower) {
															return -1;
														} else {
															return 0;
														}
													} else {
														return 0;
													}
												}
											} else {
												if ((data[index+offsets[1]]& 0xFF) < lower) {
													if ((data[index+offsets[2]]& 0xFF) < lower) {
														if ((data[index+offsets[3]]& 0xFF) < lower) {
															return -1;
														} else {
															return 0;
														}
													} else {
														return 0;
													}
												} else {
													return 0;
												}
											}
										} else {
											if ((data[index+offsets[12]]& 0xFF) < lower) {
												if ((data[index+offsets[13]]& 0xFF) < lower) {
													if ((data[index+offsets[14]]& 0xFF) < lower) {
														if ((data[index+offsets[15]]& 0xFF) < lower) {
															return -1;
														} else {
															return 0;
														}
													} else {
														return 0;
													}
												} else {
													return 0;
												}
											} else {
												return 0;
											}
										}
									} else {
										return 0;
									}
								} else {
									return 0;
								}
							} else {
								return 0;
							}
						} else {
							return 0;
						}
					} else {
						return 0;
					}
				} else {
					return 0;
				}
			} else {
				return 0;
			}
		}
	}

	public final int function6( int index ) {
		if ((data[index+offsets[0]]& 0xFF) < lower) {
			if ((data[index+offsets[1]]& 0xFF) < lower) {
				if ((data[index+offsets[2]]& 0xFF) < lower) {
					if ((data[index+offsets[3]]& 0xFF) < lower) {
						if ((data[index+offsets[4]]& 0xFF) < lower) {
							if ((data[index+offsets[6]]& 0xFF) < lower) {
								if ((data[index+offsets[7]]& 0xFF) < lower) {
									if ((data[index+offsets[8]]& 0xFF) < lower) {
										if ((data[index+offsets[9]]& 0xFF) < lower) {
											if ((data[index+offsets[10]]& 0xFF) < lower) {
												return -1;
											} else {
												if ((data[index+offsets[15]]& 0xFF) < lower) {
													return -1;
												} else {
													return 0;
												}
											}
										} else {
											if ((data[index+offsets[14]]& 0xFF) < lower) {
												if ((data[index+offsets[15]]& 0xFF) < lower) {
													return -1;
												} else {
													return 0;
												}
											} else {
												return 0;
											}
										}
									} else {
										if ((data[index+offsets[13]]& 0xFF) < lower) {
											if ((data[index+offsets[14]]& 0xFF) < lower) {
												if ((data[index+offsets[15]]& 0xFF) < lower) {
													return -1;
												} else {
													return 0;
												}
											} else {
												return 0;
											}
										} else {
											return 0;
										}
									}
								} else {
									if ((data[index+offsets[12]]& 0xFF) < lower) {
										if ((data[index+offsets[13]]& 0xFF) < lower) {
											if ((data[index+offsets[14]]& 0xFF) < lower) {
												if ((data[index+offsets[15]]& 0xFF) < lower) {
													return -1;
												} else {
													return 0;
												}
											} else {
												return 0;
											}
										} else {
											return 0;
										}
									} else {
										return 0;
									}
								}
							} else {
								if ((data[index+offsets[11]]& 0xFF) < lower) {
									if ((data[index+offsets[12]]& 0xFF) < lower) {
										if ((data[index+offsets[13]]& 0xFF) < lower) {
											if ((data[index+offsets[14]]& 0xFF) < lower) {
												if ((data[index+offsets[15]]& 0xFF) < lower) {
													return -1;
												} else {
													return 0;
												}
											} else {
												return 0;
											}
										} else {
											return 0;
										}
									} else {
										return 0;
									}
								} else {
									return 0;
								}
							}
						} else {
							if ((data[index+offsets[9]]& 0xFF) < lower) {
								if ((data[index+offsets[10]]& 0xFF) < lower) {
									if ((data[index+offsets[11]]& 0xFF) < lower) {
										if ((data[index+offsets[12]]& 0xFF) < lower) {
											if ((data[index+offsets[13]]& 0xFF) < lower) {
												if ((data[index+offsets[14]]& 0xFF) < lower) {
													if ((data[index+offsets[15]]& 0xFF) < lower) {
														return -1;
													} else {
														return 0;
													}
												} else {
													return 0;
												}
											} else {
												return 0;
											}
										} else {
											return 0;
										}
									} else {
										return 0;
									}
								} else {
									return 0;
								}
							} else {
								return 0;
							}
						}
					} else {
						if ((data[index+offsets[8]]& 0xFF) < lower) {
							if ((data[index+offsets[9]]& 0xFF) < lower) {
								if ((data[index+offsets[10]]& 0xFF) < lower) {
									if ((data[index+offsets[11]]& 0xFF) < lower) {
										if ((data[index+offsets[12]]& 0xFF) < lower) {
											if ((data[index+offsets[13]]& 0xFF) < lower) {
												if ((data[index+offsets[14]]& 0xFF) < lower) {
													if ((data[index+offsets[7]]& 0xFF) < lower) {
														if ((data[index+offsets[15]]& 0xFF) < lower) {
															return -1;
														} else {
															if ((data[index+offsets[4]]& 0xFF) < lower) {
																if ((data[index+offsets[6]]& 0xFF) < lower) {
																	return -1;
																} else {
																	return 0;
																}
															} else {
																return 0;
															}
														}
													} else {
														if ((data[index+offsets[15]]& 0xFF) < lower) {
															return -1;
														} else {
															return 0;
														}
													}
												} else {
													return 0;
												}
											} else {
												return 0;
											}
										} else {
											return 0;
										}
									} else {
										return 0;
									}
								} else {
									return 0;
								}
							} else {
								return 0;
							}
						} else {
							return 0;
						}
					}
				} else {
					if ((data[index+offsets[7]]& 0xFF) < lower) {
						if ((data[index+offsets[8]]& 0xFF) < lower) {
							if ((data[index+offsets[9]]& 0xFF) < lower) {
								if ((data[index+offsets[10]]& 0xFF) < lower) {
									if ((data[index+offsets[11]]& 0xFF) < lower) {
										if ((data[index+offsets[12]]& 0xFF) < lower) {
											if ((data[index+offsets[13]]& 0xFF) < lower) {
												if ((data[index+offsets[6]]& 0xFF) < lower) {
													if ((data[index+offsets[14]]& 0xFF) < lower) {
														if ((data[index+offsets[15]]& 0xFF) < lower) {
															return -1;
														} else {
															if ((data[index+offsets[4]]& 0xFF) < lower) {
																return -1;
															} else {
																return 0;
															}
														}
													} else {
														if ((data[index+offsets[3]]& 0xFF) < lower) {
															if ((data[index+offsets[4]]& 0xFF) < lower) {
																return -1;
															} else {
																return 0;
															}
														} else {
															return 0;
														}
													}
												} else {
													if ((data[index+offsets[14]]& 0xFF) < lower) {
														if ((data[index+offsets[15]]& 0xFF) < lower) {
															return -1;
														} else {
															return 0;
														}
													} else {
														return 0;
													}
												}
											} else {
												return 0;
											}
										} else {
											return 0;
										}
									} else {
										return 0;
									}
								} else {
									return 0;
								}
							} else {
								return 0;
							}
						} else {
							return 0;
						}
					} else {
						return 0;
					}
				}
			} else {
				if ((data[index+offsets[6]]& 0xFF) < lower) {
					if ((data[index+offsets[7]]& 0xFF) < lower) {
						if ((data[index+offsets[8]]& 0xFF) < lower) {
							if ((data[index+offsets[9]]& 0xFF) < lower) {
								if ((data[index+offsets[10]]& 0xFF) < lower) {
									if ((data[index+offsets[11]]& 0xFF) < lower) {
										if ((data[index+offsets[12]]& 0xFF) < lower) {
											if ((data[index+offsets[13]]& 0xFF) < lower) {
												if ((data[index+offsets[4]]& 0xFF) < lower) {
													if ((data[index+offsets[14]]& 0xFF) < lower) {
														return -1;
													} else {
														if ((data[index+offsets[3]]& 0xFF) < lower) {
															return -1;
														} else {
															return 0;
														}
													}
												} else {
													if ((data[index+offsets[14]]& 0xFF) < lower) {
														if ((data[index+offsets[15]]& 0xFF) < lower) {
															return -1;
														} else {
															return 0;
														}
													} else {
														return 0;
													}
												}
											} else {
												if ((data[index+offsets[2]]& 0xFF) < lower) {
													if ((data[index+offsets[3]]& 0xFF) < lower) {
														if ((data[index+offsets[4]]& 0xFF) < lower) {
															return -1;
														} else {
															return 0;
														}
													} else {
														return 0;
													}
												} else {
													return 0;
												}
											}
										} else {
											return 0;
										}
									} else {
										return 0;
									}
								} else {
									return 0;
								}
							} else {
								return 0;
							}
						} else {
							return 0;
						}
					} else {
						return 0;
					}
				} else {
					return 0;
				}
			}
		} else {
			if ((data[index+offsets[6]]& 0xFF) < lower) {
				if ((data[index+offsets[7]]& 0xFF) < lower) {
					if ((data[index+offsets[8]]& 0xFF) < lower) {
						if ((data[index+offsets[9]]& 0xFF) < lower) {
							if ((data[index+offsets[10]]& 0xFF) < lower) {
								if ((data[index+offsets[11]]& 0xFF) < lower) {
									if ((data[index+offsets[4]]& 0xFF) < lower) {
										if ((data[index+offsets[12]]& 0xFF) < lower) {
											if ((data[index+offsets[3]]& 0xFF) < lower) {
												if ((data[index+offsets[13]]& 0xFF) < lower) {
													return -1;
												} else {
													if ((data[index+offsets[2]]& 0xFF) < lower) {
														return -1;
													} else {
														return 0;
													}
												}
											} else {
												if ((data[index+offsets[13]]& 0xFF) < lower) {
													if ((data[index+offsets[14]]& 0xFF) < lower) {
														return -1;
													} else {
														return 0;
													}
												} else {
													return 0;
												}
											}
										} else {
											if ((data[index+offsets[1]]& 0xFF) < lower) {
												if ((data[index+offsets[2]]& 0xFF) < lower) {
													if ((data[index+offsets[3]]& 0xFF) < lower) {
														return -1;
													} else {
														return 0;
													}
												} else {
													return 0;
												}
											} else {
												return 0;
											}
										}
									} else {
										if ((data[index+offsets[12]]& 0xFF) < lower) {
											if ((data[index+offsets[13]]& 0xFF) < lower) {
												if ((data[index+offsets[14]]& 0xFF) < lower) {
													if ((data[index+offsets[15]]& 0xFF) < lower) {
														return -1;
													} else {
														return 0;
													}
												} else {
													return 0;
												}
											} else {
												return 0;
											}
										} else {
											return 0;
										}
									}
								} else {
									return 0;
								}
							} else {
								return 0;
							}
						} else {
							return 0;
						}
					} else {
						return 0;
					}
				} else {
					return 0;
				}
			} else {
				return 0;
			}
		}
	}

	public final int function7( int index ) {
		if ((data[index+offsets[10]]& 0xFF) > upper) {
			if ((data[index+offsets[5]]& 0xFF) > upper) {
				if ((data[index+offsets[6]]& 0xFF) > upper) {
					if ((data[index+offsets[7]]& 0xFF) > upper) {
						if ((data[index+offsets[8]]& 0xFF) > upper) {
							if ((data[index+offsets[9]]& 0xFF) > upper) {
								if ((data[index+offsets[11]]& 0xFF) > upper) {
									if ((data[index+offsets[4]]& 0xFF) > upper) {
										if ((data[index+offsets[12]]& 0xFF) > upper) {
											if ((data[index+offsets[3]]& 0xFF) > upper) {
												if ((data[index+offsets[13]]& 0xFF) > upper) {
													return 1;
												} else {
													if ((data[index+offsets[2]]& 0xFF) > upper) {
														return 1;
													} else {
														return 0;
													}
												}
											} else {
												if ((data[index+offsets[13]]& 0xFF) > upper) {
													if ((data[index+offsets[14]]& 0xFF) > upper) {
														return 1;
													} else {
														return 0;
													}
												} else {
													return 0;
												}
											}
										} else {
											if ((data[index+offsets[1]]& 0xFF) > upper) {
												if ((data[index+offsets[2]]& 0xFF) > upper) {
													if ((data[index+offsets[3]]& 0xFF) > upper) {
														return 1;
													} else {
														return 0;
													}
												} else {
													return 0;
												}
											} else {
												return 0;
											}
										}
									} else {
										if ((data[index+offsets[12]]& 0xFF) > upper) {
											if ((data[index+offsets[13]]& 0xFF) > upper) {
												if ((data[index+offsets[14]]& 0xFF) > upper) {
													if ((data[index+offsets[15]]& 0xFF) > upper) {
														return 1;
													} else {
														return 0;
													}
												} else {
													return 0;
												}
											} else {
												return 0;
											}
										} else {
											return 0;
										}
									}
								} else {
									return 0;
								}
							} else {
								return 0;
							}
						} else {
							return 0;
						}
					} else {
						return 0;
					}
				} else {
					return 0;
				}
			} else {
				return 0;
			}
		} else {
			if ((data[index+offsets[0]]& 0xFF) < lower) {
				if ((data[index+offsets[10]]& 0xFF) < lower) {
					if ((data[index+offsets[11]]& 0xFF) < lower) {
						if ((data[index+offsets[12]]& 0xFF) < lower) {
							if ((data[index+offsets[13]]& 0xFF) < lower) {
								if ((data[index+offsets[14]]& 0xFF) < lower) {
									if ((data[index+offsets[15]]& 0xFF) < lower) {
										if ((data[index+offsets[1]]& 0xFF) < lower) {
											if ((data[index+offsets[9]]& 0xFF) < lower) {
												if ((data[index+offsets[2]]& 0xFF) < lower) {
													if ((data[index+offsets[8]]& 0xFF) < lower) {
														return -1;
													} else {
														if ((data[index+offsets[3]]& 0xFF) < lower) {
															return -1;
														} else {
															return 0;
														}
													}
												} else {
													if ((data[index+offsets[7]]& 0xFF) < lower) {
														if ((data[index+offsets[8]]& 0xFF) < lower) {
															return -1;
														} else {
															return 0;
														}
													} else {
														return 0;
													}
												}
											} else {
												if ((data[index+offsets[2]]& 0xFF) < lower) {
													if ((data[index+offsets[3]]& 0xFF) < lower) {
														if ((data[index+offsets[4]]& 0xFF) < lower) {
															return -1;
														} else {
															return 0;
														}
													} else {
														return 0;
													}
												} else {
													return 0;
												}
											}
										} else {
											if ((data[index+offsets[6]]& 0xFF) < lower) {
												if ((data[index+offsets[7]]& 0xFF) < lower) {
													if ((data[index+offsets[8]]& 0xFF) < lower) {
														if ((data[index+offsets[9]]& 0xFF) < lower) {
															return -1;
														} else {
															return 0;
														}
													} else {
														return 0;
													}
												} else {
													return 0;
												}
											} else {
												return 0;
											}
										}
									} else {
										return 0;
									}
								} else {
									return 0;
								}
							} else {
								return 0;
							}
						} else {
							return 0;
						}
					} else {
						return 0;
					}
				} else {
					return 0;
				}
			} else {
				return 0;
			}
		}

	}

}
