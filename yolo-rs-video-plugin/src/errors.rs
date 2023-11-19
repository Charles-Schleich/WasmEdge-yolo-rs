
// trait FailContinue {
//     fn next() -> {}
// }


pub fn handle_failed_pointer(){
    error!("Could");
    return Err(HostFuncError::User(1));
}